 # coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import ast
import copy
import os
import glob
import sys
import pickle
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from tokenize import tokenize as py_tokenize
from util_attr import draw_graph
import astor
import torch
from torch.cuda.amp import autocast as autocast
import json
import random
import logging
import argparse
import numpy as np
from apex import amp
from mutate import mutateIfExpr2Stmt, negate_if_expr,mutateAssginAddLine,mutateLambda2func,mutateComp2For,mutateExprStmt2Assign,\
mutateAugAssgin2Assgin,mutateAssert,mutateIfStmt,mutateWhileStmt,mutateForStmt,mutateWith2Try,mutateZipScope,mutateZipAssgin,\
    generate_random_name,findIdentifier

from io import open, BytesIO
from itertools import cycle
import torch.nn as nn
from AttrModel import AttrSeq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from fuzzywuzzy import fuzz
import re
import multiprocessing
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              AutoTokenizer,AutoModelForCausalLM)
from AttrRoberta import AttrRobertaModel
import pandas as pd
cpu_cont = 32
lits = json.load(open("literals.json"))
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = None

from nltk.translate.bleu_score import sentence_bleu
end_ids=[]
def visit(node, nodes, seed_nodes,pindex):
    if hasattr(node, 'name'):
        name = node.name
    elif hasattr(node, 'id'):
        name = node.id
    else:
        name = str(type(node).__name__)
    if type(node) == ast.Constant:
        name = '{}:{}'.format(type(node.value).__name__, node.value)
    if type(node) == ast.Attribute:
        name = '{}:{}'.format(type(node).__name__, node.attr)
    if type(node) == ast.FunctionDef:
        name = '{}:{}'.format('def', node.name)
    if type(node) == ast.Store:
        return
    if type(node) == ast.Name or type(node) == ast.Constant or type(node) == ast.FunctionDef:
        seed_nodes.append(node)
    index = len(nodes)
    nodes.append(index)
    for n in ast.iter_child_nodes(node):
        n.parent = node
        visit(n, nodes, seed_nodes,index)
    return seed_nodes
def calculate_bleu_score(generated_text, reference_text,tokenizer):
    # Convert the strings to lists of tokens
    # generated_tokens = generated_text.split()
    # reference_tokens = reference_text.split()
    # generated_tokens = [x for x in py_tokenizer.generate_tokens(BytesIO(generated_text.encode('utf-8')).readline)]
    # reference_tokens = [x for x in py_tokenizer.generate_tokens(BytesIO(reference_text.encode('utf-8')).readline)]
    generated_tokens = [x.replace('Ġ','') for x in tokenizer.tokenize(generated_text)]
    reference_tokens = [x.replace('Ġ','') for x in tokenizer.tokenize(reference_text)]
    # Compute the BLEU score
    weights = (0.48, 0.24, 0.16, 0.12)  # Use 4-gram BLEU score with uniform weights
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, weights=weights)

    return bleu_score
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 pos=None,
                 action=None,
                 score = None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.pos = pos
        self.action = action
        self.score = score

def read_examples_str(dict_list):
    """Read examples from filename."""
    examples=[]
    num = 0
    for idx, js in enumerate(dict_list):
        # if idx==100:
        #     return examples

        inputs = js["input"]
        outputs = js["gt"]
        pos = js["pos"]
        action = js["action"]
        if 'id' in js:
            idx = js['id']
        if 'score' in js:
            score = js['score']
        examples.append(
            Example(
                idx=idx,
                source=inputs,
                target=outputs,
                pos=pos,
                action=action,
            )
        )
    return examples
def read_examples(filename,mutate=False):
    """Read examples from filename."""
    examples=[]
    num = 0
    with open(filename,encoding="utf-8") as f:
        if mutate:
            for idx, line in tqdm(enumerate(f),desc='read examples'):
                # if idx==100:
                #     return examples
                try:
                    js = json.loads(line)
                except:
                    continue
                inputs = js["input"].replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = js["gt"]
                action = js["action"]
                if 'id' in js:
                    idx = js['id']
                examples.append(
                    Example(
                        idx=idx,
                        source=inputs,
                        target=outputs,
                        action=action
                    )
                )
        if 'short' in filename:
            js = json.load(f)
            for ct, item in enumerate(js['inputs']):
                inputs = item["input"].replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = item['gt']
                idx = str(js['id'].replace('/','-')) + '-' + str(ct)
                examples.append(
                    Example(
                        idx=idx,
                        source=inputs,
                        target=outputs,
                    )
                )
        if 'long' in filename:
            js = json.load(f)
            for ct, item in enumerate(js['inputs']):
                inputs = item["input"].replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = item['gt']
                idx = str(js['id']) + '-' + str(ct)
                examples.append(
                    Example(
                        idx=idx,
                        source=inputs,
                        target=outputs,
                    )
                )
        else:
            for idx, line in enumerate(f):
                if ".txt" in filename:
                    inputs=line.strip().replace("<EOL>","</s>").split()
                    inputs=inputs[1:]
                    inputs=" ".join(inputs)
                    outputs=[]
                    examples.append(
                        Example(
                            idx=idx,
                            source=inputs,
                            target=outputs,
                        )
                    )
                else:
                    js=json.loads(line)
                    inputs=js["text"].replace("<EOL>","</s>").split()
                    inputs=inputs[1:]
                    inputs=" ".join(inputs)
                    outputs=js["gt"]
                    if 'id' in js:
                        idx = js['id']
                    examples.append(
                        Example(
                                idx = idx,
                                source = inputs,
                                target = outputs,
                                )
                    )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids

def post_process(code):
    code = code.replace("<string","<STR_LIT").replace("<number","<NUM_LIT").replace("<char","<CHAR_LIT")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

def tokenize(item):
    source, max_length, tokenizer = item
    source_tokens = [x for x in tokenizer.tokenize(source)]
    source_tokens = source_tokens
    if len(source_tokens)>max_length:
        print('over length')
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens[-max_length:])
    # padding_length = max_length - len(source_ids)
    # source_ids+=[tokenizer.pad_token_id]*padding_length
    return source_tokens,source_ids

def convert_examples_to_features(examples, tokenizer, args,pool=None,stage=None):
    features = []
    if stage=="train":
        max_length = args.max_source_length+args.max_target_length
    else:
        max_length = args.max_source_length
    sources = [(x.source,max_length,tokenizer) for x in examples]
    if pool is not None:
        tokenize_tokens = pool.map(tokenize,tqdm(sources,total=len(sources)))
    else:
        tokenize_tokens = [tokenize(x) for x in sources]
    for example_index, (source_tokens,source_ids) in enumerate(tokenize_tokens):
        #source
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
            )
        )
    return features

def convert_examples_to_features_long(examples, tokenizer, args,pool=None,stage=None):
    features = []
    if stage=="train":
        max_length = args.max_source_length+args.max_target_length
    else:
        max_length = args.max_source_length
    sources = [(x.source,max_length,tokenizer) for x in examples]
    if pool is not None:
        tokenize_tokens = pool.map(tokenize,tqdm(sources,total=len(sources)))
    else:
        tokenize_tokens = [tokenize(x) for x in sources]
    for example_index, (source_tokens,source_ids) in enumerate(tokenize_tokens):
        #source
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
            )
        )
    return features



def set_seed(seed=996):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale
        res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def generate_statement(model,input,end_ids):
    global device
    stop_num = 0
    output_seq = []
    context_length = input.size(1)
    num_batch = input.size(0)
    # state_start = np.array([0] * num_batch, dtype=int)
    state_end = np.array([0] * num_batch, dtype=int)
    output_logits = []
    model.eval()
    with torch.no_grad():
        outputs = model(input)
    logits = outputs['logits'].detach()
    shift_logits = logits[..., -1, :].detach()
    input_logits = torch.softmax(shift_logits, dim=-1)
    shift_pred = input_logits.argmax(-1)
    input_ids = shift_pred.unsqueeze(1)
    output_seq.append(input_ids)
    output_logits.append(input_logits.unsqueeze(1).detach())
    input = input_ids
    context = outputs['past_key_values']
    while stop_num<100:
        length = input.size(1)+context_length
        with torch.no_grad():
            outputs=model(input,attention_mask=torch.ones((num_batch, length), dtype=torch.uint8,device=device),past_key_values=context)
        # outputs = model(input, attention_mask=torch.ones((num_batch, length), dtype=torch.uint8, device='cpu'),
        #                 past_key_values=context)
        logits = outputs['logits'].clone().detach()
        del outputs
        shift_logits = logits[...,-1, :].detach()
        input_logits=torch.softmax(shift_logits,dim=-1)
        shift_pred = input_logits.argmax(-1)
        input_ids = shift_pred.unsqueeze(1)
        output_seq.append(input_ids)
        output_logits.append(input_logits.unsqueeze(1).detach())
        input = torch.cat([input,input_ids],dim=-1).detach()
        stop_num+=1
        for i in range(num_batch):
            if shift_pred[i] in end_ids:
                state_end[i]=1
        if state_end.sum() == num_batch:
            break
    del context
    output_seq=torch.cat(output_seq,dim=1).detach()
    output_logits=torch.cat(output_logits,dim=1)
    # for i in range(output_seq.shape[0]):
    #     print(tokenizer.decode(output_seq[i]))
    return output_seq,output_logits


def get_start_line(src_code):
    num = 0
    count_dict = {}
    code = ''.join(src_code)
    token_gen = py_tokenize(BytesIO(bytes(code, "utf8")).readline)
    out_tokens = []
    prev_eol = False
    start_line = 0
    for toknum, tokval, start, end, line in token_gen:
        # if start==pos and arg_switch==0:
        #     # break
        #     None
        # elif start==pos and arg_switch==1:
        #     break_flag=1
        tokval = " ".join(tokval.split())
        # if arg_switch==1:
        #     if tokval!=dst_code[start[0]-1][start[1]:end[1]] and break_flag==1:
        #         # break
        #         None

        if toknum == STRING:
            add_token = process_string(tokval)
            out_tokens.append(add_token)
            prev_eol = False
        elif toknum == NUMBER:
            if tokval in lits['num']:
                out_tokens.append(f"<NUM_LIT:{tokval}>")
            else:
                out_tokens.append(f"<NUM_LIT>")
            prev_eol = False
        elif toknum in [NEWLINE, NL]:
            if not prev_eol:
                out_tokens.append("<EOL>")
                prev_eol = True
        elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
            continue
        else:
            out_tokens.append(tokval)
            prev_eol = False
        if len(out_tokens) > 900:
            start_line = end[0]
            break
    return start_line

def mask_code(start,end,src_code,tokenizer,tgt_line,new_line):
    if start[0] == end[0]:
        tokens = tokenizer.tokenize(src_code[start[0]-1][start[1]:end[1]])
        mask_code = src_code[start[0]-1][:start[1]]+tokenizer.unk_token*len(tokens)+src_code[start[0]-1][end[1]:]
        return ''.join(src_code[tgt_line-100 if tgt_line-100>= 0 else 0:start[0]-1]+[mask_code]+src_code[start[0]:tgt_line])+new_line
    else:
        mask_code = src_code[start[0]-1][start[1]:]
        for i in range(start[0],end[0]):
            mask_code+=src_code[i]
        mask_code+=src_code[end[0]-1][:end[1]]
        tokens = tokenizer.tokenize(mask_code)
        mask_code = src_code[start[0]-1][:start[1]]+tokenizer.unk_token*len(tokens)+src_code[end[0]-1][end[1]:]
        return ''.join(src_code[tgt_line-100 if tgt_line-100>= 0 else 0:start[0]-1]+[mask_code]+src_code[end[0]:tgt_line])+new_line
def find_adv_seed(src_code,tgt_line,seed_nodes, tokenizer, model, args,src_target,src_text,src_logits,new_line,target_label):
    global device
    nodes = []
    code = ''.join(src_code)
    for node in seed_nodes:
        if node.end_lineno <= tgt_line and node.lineno > tgt_line-100:
            nodes.append(node)
    mask_datas=[]
    for idx,node in enumerate(nodes):
        if node.lineno == node.end_lineno:
            start_node = (node.lineno,node.col_offset)
            end_node = (node.lineno,node.end_col_offset)
        else:
            start_node = (node.lineno,node.col_offset)
            end_node = (node.end_lineno,node.end_col_offset)
        input_str = mask_code(start_node,end_node,src_code,tokenizer,tgt_line,new_line)
        mask_datas.append({
            'idx': idx,
            'input': input_str,
            'gt': src_text,
            'pos': (start_node, end_node),
            'action': '',
        })
    with torch.no_grad():
        eval_examples = read_examples_str(mask_datas)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        # all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        # eval_data = TensorDataset(all_source_ids)
        # # Calculate bleu
        # eval_sampler = SequentialSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # if not os.path.exists('single_vec_test_'+args.lang):
        #     os.mkdir('single_vec_test_'+args.lang)
        model.eval()
        p = []
        log = []
        idx_ = 0
        for batch in eval_features:
            # source_ids = torch.tensor([batch.source_ids], dtype=torch.long).to('cuda')
            source_ids = torch.tensor([batch.source_ids], dtype=torch.long).to(device)
            preds, logits = generate_statement(model, source_ids, end_ids)
            preds = preds.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            for pred, logit in zip(preds, logits):
                text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                text = text[:text.find('\n')].strip()
                if text == src_text:
                    log.append((True, text, pred, logit))
                else:
                    log.append((False, text, pred, logit))
    result = []
    assert len(log) == len(nodes)
    for i_node,logit in enumerate(log):
        same = logit[0]
        score = 0
        flag_eos = False
        pred = logit[2]

        # for i in range(len(src_target)):
        #     if i < len(pred) and flag_eos == False:
        #         if pred[i] == 2:
        #             flag_eos = True
        #         if src_target[i] == pred[i]:
        #             score += src_logits[i][src_target[i]] - logit[3][i][src_target[i]]
        #         else:
        #             score += src_logits[i][src_target[i]] - logit[3][i][src_target[i]] + logit[3][i][pred[i]] - \
        #                      src_logits[i][pred[i]]
        #     else:
        #         score += src_logits[i][src_target[i]]
        i=0
        score += src_logits[i][src_target[i]] - logit[3][i][src_target[i]] + logit[3][i][target_label] - \
                     src_logits[i][pred[i]]
        # if src_target[i] == pred[i]:
        #     score += src_logits[i][src_target[i]] - logit[3][i][src_target[i]]
        # else:
        #     score += src_logits[i][src_target[i]] - logit[3][i][src_target[i]] + logit[3][i][pred[i]] - \
        #              src_logits[i][pred[i]]
        score = float(score)
        result.append((same, score, logit[1],nodes[i_node],logit[3][i]))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    result_final = result
    # for i in result:
    #     if not i[0] and i not in result_final:
    #         result_final.append(i)
    return result_final
def apply_action(src_code,actions,node,tgt_line,next_line,score):
    mutate_datas=[]
    for action in actions:
        line_add = 0
        replace_flag = False
        if action[0] == 'replaceName':
            replace_flag=True
            old_name = action[1]
            new_name = action[3]
            token_gen = py_tokenize(BytesIO(bytes(''.join(src_code), "utf8")).readline)
            replace_pos = []
            temp_dot=''
            for toknum, tokval, start, end, line in token_gen:
                if tokval == old_name:
                    if temp_dot!='.':
                        replace_pos.append((start,end))
                temp_dot = tokval

            new_src_code = copy.deepcopy(src_code)
            replace_dict = {}
            for pos in replace_pos:
                start = pos[0]
                start_line = start[0]
                if start_line not in replace_dict.keys():
                    replace_dict[start_line] = [pos]
                else:
                    replace_dict[start_line].append(pos)
            for line_action in replace_dict.keys():
                if len(replace_dict[line_action]) == 1:
                    start = replace_dict[line_action][0][0][1]
                    end = replace_dict[line_action][0][1][1]
                    new_src_code[line_action-1] = new_src_code[line_action-1][:start] + new_name + new_src_code[line_action-1][end:]
                else:
                    len_add=0
                    for pos in replace_dict[line_action]:
                        start = pos[0][1]
                        end = pos[1][1]
                        new_src_code[line_action-1] = new_src_code[line_action-1][:start+len_add] + new_name + new_src_code[line_action-1][end+len_add:]
                        len_add += len(new_name) - len(old_name)
                if line_action-1 == tgt_line:
                    if len(replace_dict[line_action]) == 1:
                        start = replace_dict[line_action][0][0][1]
                        end = replace_dict[line_action][0][1][1]
                        if start < len(next_line):
                            next_line = next_line[:start] + new_name + next_line[end:]
                    else:
                        len_add = 0
                        for pos in replace_dict[line_action]:
                            start = pos[0][1]
                            end = pos[1][1]
                            if start+len_add >= len(next_line):
                                break
                            next_line = next_line[:start + len_add] + new_name + next_line[end + len_add:]
                            len_add += len(new_name) - len(old_name)
            
        elif action[0] == 'IfExpr2Stmt':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_line:]
            line_add = 1 - len(re.findall('\n', src_code[act_line - 1]))
        elif action[0] == 'negate_if_expr':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(
                re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'AssginAddLine':
            act_line = action[2][0]
            new_line = src_code[act_line - 1]
            enter_start = re.findall('\n', new_line)
            comment = ''
            if new_line.find('#') != -1:
                new_line_ = new_line[:new_line.find('#')]
                comment = new_line[new_line.find('#'):]
                new_line = new_line_
            if '(\n' in new_line and '\n)' in new_line:
                new_line = new_line.replace('(\n', '(')
                new_line = new_line.replace('\n)', ')')
            elif '(' in new_line and ')' in new_line:
                new_line = new_line.replace('(', '(\n')
                new_line = new_line.replace(')', '\n)')
            if '{\n' in new_line and '\n}' in new_line:
                new_line = new_line.replace('{\n', '{')
                new_line = new_line.replace('\n}', '}')
            elif '{' in new_line and '}' in new_line:
                new_line = new_line.replace('{', '{\n')
                new_line = new_line.replace('}', '\n}')
            if '[\n' in new_line and '\n]' in new_line:
                new_line = new_line.replace('[\n', '[')
                new_line = new_line.replace('\n]', ']')
            elif '[' in new_line and ']' in new_line:
                new_line = new_line.replace('[', '[\n')
                new_line = new_line.replace(']', '\n]')
            enter_stop = re.findall('\n', new_line)
            # line_add = len(enter_stop) - len(enter_start)
            line_add = 0
            new_src_code = src_code[:act_line - 1] + [new_line + comment] + src_code[act_line:]
        elif action[0] == 'Lambda2Func':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            if action[4][0] == action[4][1]:
                repstr = src_code[action[4][0] - 1][:action[4][-2]] + action[5] + src_code[action[4][0] - 1][
                                                                                  action[4][-1]:]
                ctx_ = src_code[act_line - 1:]
                ctx_[action[4][0] - act_line] = repstr
                new_src_code = src_code[:act_line - 1] + [new_line] + ctx_
            else:
                repstr = src_code[action[4][0] - 1][:action[4][-2]] + action[5] + src_code[action[4][1] - 1][
                                                                                  action[4][3]:]
                ctx_ = src_code[act_line - 1:]
                ctx = [repstr] + src_code[action[4][1]:]
                # src_code[:act_line - 1] + [new_line] + [
                # ] + src_code[action[4][1]:]
                new_src_code = src_code[:act_line - 1] + [new_line] + ctx
            line_add = 1 + len(re.findall('\n', repstr)) - len(
                re.findall('\n', ''.join(src_code[action[4][0] - 1:action[4][1]])))
        elif action[0] == 'Comp2For':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_line:]
            line_add = 1 - len(re.findall('\n', src_code[act_line - 1]))
        elif action[0] == 'ExprStmt2Assign':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_src_code = src_code[:act_line - 1] + [
                src_code[act_line - 1][:st] + action[3] + src_code[act_line - 1][st:]] + src_code[act_line:]
            line_add = 0
        elif action[0] == 'AugAssgin2Assign':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_line:]
            line_add = 1 - len(re.findall('\n', src_code[act_line - 1]))
        elif action[0] == 'Assert2If':
            act_line = action[2][0]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_line:]
            line_add = 1 - len(re.findall('\n', src_code[act_line - 1]))
        elif action[0] == 'IfStmt2IfStmt':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'WhileStmt':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]

            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'ForStmt':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'With2Try':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'Scope2Func':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        elif action[0] == 'ZipAssgin':
            act_line = action[2][0]
            act_end = action[2][1]
            st = re.findall('\S', src_code[act_line - 1])
            st = src_code[act_line - 1].find(st[0])
            new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                     :-len(src_code[act_line - 1][:st])]
            new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
            line_add = 1 - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
        # elif action[0] == 'GarbageCode':
        #     if action[1] == 'all':
        #         act_line = random.choice(add_code)
        #         while act_line - 1 >= tg_line:
        #             act_line = random.choice(add_code)
        #         near = 10000
        #         scope = {}
        #         for key, value in scope_list.items():
        #             if act_line > value['##scope##'][0] and act_line < value['##scope##'][1]:
        #                 if value['##scope##'][1] - value['##scope##'][0] < near:
        #                     scope = value
        #         ass_list = set()
        #         use_list = set()
        #         for key in scope.keys():
        #             if key == "##scope##":
        #                 continue
        #             for idx, _ in scope[key]['assign']:
        #                 if idx < act_line:
        #                     ass_list.add(key)
        #         for key in scope.keys():
        #             min_use = 10000
        #             min_ass = 10000
        #             if key == "##scope##":
        #                 continue
        #             for idx, _ in scope[key]['use']:
        #                 if idx > act_line:
        #                     if idx < min_use:
        #                         min_use = idx
        #             for idx, _ in scope[key]['assign']:
        #                 if idx > act_line:
        #                     if idx < min_ass:
        #                         min_ass = idx
        #             if min_ass < min_use:
        #                 use_list.add(key)
        #         ass_list = list(ass_list)
        #         use_list = list(use_list)
        #         if len(ass_list) > 0 and len(use_list) > 0:
        #             new_line = astor.to_source(ast.If(
        #                 test=ast.Compare(ops=[ast.Eq()], left=ast.Name(id=ass_list[0], ctx=ast.Load()),
        #                                  comparators=[ast.Name(id=ass_list[0], ctx=ast.Load())]), body=[
        #                     ast.Assign(targets=[ast.Name(id=use_list[0], ctx=ast.Load())],
        #                                value=ast.Name(id=ass_list[-1], ctx=ast.Load()))], orelse=[]))
        #         elif len(use_list) > 0:
        #             new_line = astor.to_source(ast.If(test=ast.Constant(value='True'), body=[
        #                 ast.Assign(targets=[ast.Name(id=use_list[0], ctx=ast.Load())],
        #                            value=ast.Constant(value=random.randint(0, 99)))], orelse=[]))
        #         elif len(ass_list) > 0:
        #             new_line = astor.to_source(ast.If(test=ast.Compare(ops=[ast.Eq()],
        #                                                                left=ast.Name(id=ass_list[0], ctx=ast.Load()),
        #                                                                comparators=[
        #                                                                    ast.Name(id=ass_list[0],
        #                                                                             ctx=ast.Load())]),
        #                                               body=[ast.Pass],
        #                                               orelse=[]))
        #         else:
        #             new_line = astor.to_source(ast.If(test=ast.Constant(value='True'), body=[ast.Pass], orelse=[]))
        #         up_line = act_line - 1
        #         while len(re.findall('\S', src_code[up_line])) == 0:
        #             up_line = up_line - 1
        #         st = re.findall('\S', src_code[up_line])
        #         new_enter = new_line.find('\n')
        #         down_line = act_line
        #         while len(re.findall('\S', src_code[down_line])) == 0:
        #             down_line = down_line + 1
        #         if st[-1] != ':':
        #             st = src_code[up_line].find(st[0])
        #             new_src_code = src_code[:act_line] + [
        #                 src_code[up_line][:st] + new_line[:new_enter + 1] + src_code[up_line][:st] + new_line[
        #                                                                                              new_enter + 1:]] + src_code[
        #                                                                                                                 act_line:]
        #         else:
        #             st = re.findall('\S', src_code[down_line])
        #             st = src_code[down_line].find(st[0])
        #             new_src_code = src_code[:act_line] + [
        #                 src_code[down_line][:st] + new_line[:new_enter + 1] + src_code[down_line][:st] + new_line[
        #                                                                                                  new_enter + 1:]] + src_code[
        #                                                                                                                     act_line:]
        else:
            continue
        # continue
        code = ''.join(new_src_code[tgt_line - 100 if tgt_line-100>= 0 else 0:tgt_line + line_add])+next_line
        src_str = code
        start_node = (node.lineno, node.col_offset)
        end_node = (node.end_lineno, node.end_col_offset)
        mutate_datas.append({
            'idx': 0,
            'input': src_str,
            'lines':new_src_code[tgt_line - 100 if tgt_line-100>= 0 else 0:tgt_line + line_add+1],
            'next_line':next_line,
            'gt': '',
            'pos': (start_node, end_node),
            'action': action,
            'score': score,
            'add_line':line_add,
        })

    return mutate_datas


def mutateName(nodes,name_set,scope):
    action_list=[]
    for node in nodes:
        if node.id!='self':
            new_name = generate_random_name(name_set)
            action_list.append(("replaceName",node.id,[node.lineno],new_name))
    # if len(action_list) >100:
    #     action_list = random.sample(action_list,100)
    return action_list


def mutate(code, tgt_line, new_seeds,name_set,scope_list,new_line):
    input_dataset = []
    for seed in new_seeds:
        node = seed[3]
        score = seed[1]
        if type(node) == ast.Name:
            input_dataset += apply_action(code, mutateName([node],name_set,scope_list), node, tgt_line, new_line,score)
        if type(node) == ast.IfExp:
            input_dataset+=apply_action(code,mutateIfExpr2Stmt([node]),node,tgt_line,new_line,score)
            input_dataset += apply_action(code, negate_if_expr([node]), node, tgt_line, new_line,score)
        if type(node) == ast.Assign:
            if type(node.value) == ast.List or type(node.value) == ast.Tuple or type(node.value) == ast.Set or type(
                    node.value) == ast.Dict or type(node.value) == ast.Call:
                input_dataset += apply_action(code, mutateAssginAddLine([node]), node, tgt_line, new_line,score)
        if type(node) == ast.Lambda:
            input_dataset += apply_action(code, mutateLambda2func([node],name_set), node, tgt_line, new_line,score)
        if type(node) == ast.ListComp or type(node) == ast.SetComp or type(node) == ast.DictComp:
            input_dataset += apply_action(code, mutateComp2For([node]), node, tgt_line, new_line,score)
        if type(node) == ast.Expr and type(node.parent) != ast.Assign and type(node.parent) != ast.AugAssign and type(
                node.parent) != ast.Module:
            input_dataset += apply_action(code, mutateExprStmt2Assign([node],name_set), node, tgt_line, new_line,score)
        if type(node) == ast.AugAssign:
            input_dataset += apply_action(code, mutateAugAssgin2Assgin([node]), node, tgt_line, new_line,score)
        if type(node) == ast.Assert:
            input_dataset += apply_action(code, mutateAssert([node]), node, tgt_line, new_line,score)
        if type(node) == ast.If:
            input_dataset += apply_action(code, mutateIfStmt([node]), node, tgt_line, new_line,score)
        if type(node) == ast.While:
            input_dataset += apply_action(code, mutateWhileStmt([node]), node, tgt_line, new_line,score)
        if type(node) == ast.For:
            input_dataset += apply_action(code, mutateForStmt([node],name_set), node, tgt_line, new_line,score)
        if type(node) == ast.With:
            input_dataset += apply_action(code, mutateWith2Try([node]), node, tgt_line, new_line,score)
        if hasattr(node, 'body') and isinstance(node.body, list):
            if len(node.body) > 3:
                start = random.randint(0, len(node.body))
                if start + 3 < len(node.body):
                    end = random.randint(start+2, start + 3)
                else:
                    end = random.randint(start, len(node.body))
                sub_scope = node.body[start:end]
                flag=0
                if len(sub_scope) > 1:
                    for sub_node in sub_scope:
                        if type(sub_node) != ast.Assign:
                            flag=1
                        else:
                            for store in sub_node.targets:
                                if type(store) != ast.Name:
                                    flag=1
                    if flag == 0 and node.body[end-1].end_lineno-node.body[start].lineno<20 :
                        input_dataset += apply_action(code, mutateZipScope([node.body[start:end]],name_set), node, tgt_line, new_line,score)
        if hasattr(node, 'body') and isinstance(node.body, list):
            # for sub_node in node.body:
            #     add_GarbageCode.append(sub_node.lineno)
            #     input_dataset += apply_action(code, mutateIfExpr2Stmt(node), node, tgt_line)
            i = 0
            while i < len(node.body):
                sub_list = []
                used_name = set()
                flag_use = 0
                if type(node.body[i]) == ast.Assign and len(node.body[i].targets) == 1:
                    sub_list.append(node.body[i])
                    if type(node.body[i].targets[0]) == ast.Name:
                        used_name.add(node.body[i].targets[0].id)
                    if type(node.body[i].targets[0]) == ast.Attribute:
                        used_name.add(astor.to_source(node.body[i].targets[0]))
                    while i+1 < len(node.body) and type(node.body[i+1]) == ast.Assign and len(node.body[i].targets) == 1:
                        for _sub_sub in ast.walk(node.body[i+1]):
                            if type(_sub_sub) == ast.Name:
                                if _sub_sub.id in used_name:
                                    flag_use=1
                                    break
                            if type(_sub_sub) == ast.Attribute:
                                if astor.to_source(_sub_sub) in used_name:
                                    flag_use = 1
                                    break
                        if flag_use == 1:
                            flag_use = 0
                            break
                        sub_list.append(node.body[i+1])
                        i += 1
                i += 1
            # if len(sub_list) > 1:
                # input_dataset += apply_action(code, mutateZipAssgin([node]), node, tgt_line,new_line, new_line,score)
    return input_dataset

def find_parent(seed_nodes):
    i = 0
    new_seed_nodes = []
    while i < len(seed_nodes):
        node = seed_nodes[i]
        if hasattr(node, 'parent'):
            if type(node) != ast.FunctionDef and type(node) != ast.Module:
                if node.parent not in seed_nodes and hasattr(node.parent, 'lineno') and node.parent not in new_seed_nodes:
                    new_seed_nodes.append(node.parent)
            i += 1
            continue
        i += 1
    return new_seed_nodes

def reset_tree(root):
    for node in ast.walk(root):
        if hasattr(node, 'lineno'):
            if hasattr(node,'score'):
                delattr(node,'score')
    return

def map_token2node(root, code, tokenizer,score, tgt_line):
    start_line  = tgt_line - 100 if tgt_line-100>= 0 else 0
    token_map = [[]]
    row = start_line+1
    col = 0

    for score_idx,token in enumerate(tokenizer.tokenize(code)):
        if 'Ċ' in token:
            tokens = token.split('Ċ')
            for i_tok,_token in enumerate(tokens):
                token_map[-1].append((_token, row, col, col+ len(_token),0))
                col += len(_token)
                if i_tok != len(tokens)-1:
                    token_map.append([])
                    row+=1
                    col=0
        else:
            while token.startswith('Ġ'):
                token_map[-1].append(('Ġ', row, col, col+1,0))
                col+=1
                token = token[1:]
            token_map[-1].append((token, row, col ,col+len(token), score[score_idx]))
            col += len(token)
    for node in ast.walk(root):
        if hasattr(node, 'lineno'):
            if node.lineno >= start_line and node.end_lineno <= tgt_line:
                if type(node) == ast.ImportFrom or type(node) == ast.Import:
                    for api in node.names:
                        for token_idx, token in enumerate(token_map[api.lineno - start_line-1]):
                            if token[2] >= api.col_offset and token[3] <= api.end_col_offset:
                                if hasattr(api, 'score'):
                                    api.score += token[4]
                                else:
                                    api.score = token[4]
                if node.lineno == node.end_lineno:
                    for token_idx, token in enumerate(token_map[node.lineno - start_line-1]):
                        if token[2] >= node.col_offset and token[3] <= node.end_col_offset:
                            if hasattr(node, 'score'):
                                node.score += token[4]
                            else:
                                node.score = token[4]
                else:
                    for l in range(node.lineno, node.end_lineno+1):
                        if l == node.lineno:
                            for token_idx, token in enumerate(token_map[l - start_line-1]):
                                if token[2] >= node.col_offset:
                                    if hasattr(node, 'score'):
                                        node.score += token[4]
                                    else:
                                        node.score = token[4]
                        elif l == node.end_lineno:
                            for token_idx, token in enumerate(token_map[l - start_line-1]):
                                if token[3] <= node.end_col_offset:
                                    if hasattr(node, 'score'):
                                        node.score += token[4]
                                    else:
                                        node.score = token[4]
                        else:
                            for token_idx, token in enumerate(token_map[l - start_line-1]):
                                if hasattr(node, 'score'):
                                    node.score += token[4]
                                else:
                                    node.score = token[4]



    return

def find_top10_identifier(tree, tgt_line, tokenizer):
    top10_identifier = []
    token_dict = {}
    start_line = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            if node.lineno > start_line and node.end_lineno <= tgt_line:
                if type(node) == ast.Name and not hasattr(node, 'attr'):
                    if hasattr(node, 'parent'):
                        if type(node.parent) == ast.Call:
                            if node == node.parent.func:
                                continue
                    if node.id in token_dict.keys():
                        token_dict[node.id].append(node.score)
                    else:
                        if hasattr(node, 'score'):
                            token_dict[node.id] = [node.score]
                        else:
                            print('error identifier')
    for key in token_dict.keys():
        token_dict[key] = sum(token_dict[key])/len(token_dict[key])
    token_dict = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)

    return token_dict

def find_top10_expr(tree, tgt_line, tokenizer):
    top10_expr = []
    token_dict = {}
    start_line = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            if node.lineno > start_line and node.end_lineno <= tgt_line:

                if type(node) == ast.Assign:
                    if type(node.value) == ast.List or type(node.value) == ast.Tuple or type(
                            node.value) == ast.Set or type(
                            node.value) == ast.Dict or type(node.value) == ast.Call:
                        if hasattr(node, 'score'):
                            if astor.to_source(node) in token_dict.keys():
                                token_dict[astor.to_source(node)]['node'].append(node)
                                token_dict[astor.to_source(node)]['score'].append(node.score)
                            else:
                                token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                        else:
                            print('error')
                if type(node) == ast.IfExp:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.Lambda:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.ListComp:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.DictComp:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.SetComp:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')

    for key in token_dict.keys():
        token_dict[key]['score'] = sum(token_dict[key]['score'])/len(token_dict[key]['score'])
    token_dict = sorted(token_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return token_dict

def find_top10_simple_stmt(tree, tgt_line, tokenizer):
    top10_simple_stmt = []
    token_dict = {}
    start_line = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            if node.lineno > start_line and node.end_lineno <= tgt_line:
                if type(node) == ast.Expr and type(node.parent) != ast.Assign:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.AugAssign:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node': [node], 'score': [node.score]}
                    else:
                        print('error')
                if type(node) == ast.Assert:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node': [node], 'score': [node.score]}
                    else:
                        print('error')
    for key in token_dict.keys():
        token_dict[key]['score'] = sum(token_dict[key]['score'])/len(token_dict[key]['score'])
    token_dict = sorted(token_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return token_dict

def find_top10_comp_stmt(tree, tgt_line, tokenizer):
    top10_comp_stmt = []
    token_dict = {}
    start_line = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            if node.lineno > start_line and node.end_lineno <= tgt_line:
                if type(node) == ast.If:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.For:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.While:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.With:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
    for key in token_dict.keys():
        token_dict[key]['score'] = sum(token_dict[key]['score'])/len(token_dict[key]['score'])
    token_dict = sorted(token_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return token_dict

def find_top10_api(tree, tgt_line, tokenizer):
    top10_api = []
    token_dict = {}
    start_line = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            if node.lineno > start_line and node.end_lineno <= tgt_line:
                # if type(node) == ast.Call:
                #     if hasattr(node, 'score'):
                #         if astor.to_source(node) in token_dict.keys():
                #             token_dict[astor.to_source(node)]['node'].append(node)
                #             token_dict[astor.to_source(node)]['score'].append(node.score)
                #         else:
                #             token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                #     else:
                #         print('error')
                if type(node) == ast.Attribute:
                    if hasattr(node, 'score'):
                        if astor.to_source(node) in token_dict.keys():
                            token_dict[astor.to_source(node)]['node'].append(node)
                            token_dict[astor.to_source(node)]['score'].append(node.score)
                        else:
                            token_dict[astor.to_source(node)] = {'node':[node], 'score':[node.score]}
                    else:
                        print('error')
                if type(node) == ast.ImportFrom or type(node) == ast.Import:
                    for api in node.names:
                        if api.asname:
                            if hasattr(api, 'score'):
                                if api.asname in token_dict.keys():
                                    token_dict[api.asname]['node'].append(api)
                                    token_dict[api.asname]['score'].append(api.score)
                                else:
                                    token_dict[api.asname] = {'node':[api], 'score':[api.score]}
                            else:
                                print('error')
                        else:
                            if hasattr(api, 'score'):
                                if api.name in token_dict.keys():
                                    token_dict[api.name]['node'].append(api)
                                    token_dict[api.name]['score'].append(api.score)
                                else:
                                    token_dict[api.name] = {'node':[api], 'score':[api.score]}
                            else:
                                print('error')
    for key in token_dict.keys():
        token_dict[key]['score'] = sum(token_dict[key]['score'])/len(token_dict[key]['score'])
    token_dict = sorted(token_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return token_dict

def fix_tree(tree,action,add_line,src_node):
    if action[0] == 'replaceName':
        for node in ast.walk(tree):
            if hasattr(node, 'id'):
                if node.id == action[1]:
                    node.id = action[3]
    if action[0] == 'AssginAddLine':
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno > src_node.lineno:
                    node.lineno += add_line
                if node.end_lineno > src_node.lineno:
                    node.end_lineno += add_line
    if action[0] == 'Comp2For':
        parent = src_node.parent
        while not hasattr(parent,'body'):
            parent = parent.parent
        for i in range(len(parent.body)):
            if parent.body[i] == src_node:
                parent.body[i] = action[5][0]
                parent.body.insert(i+1,action[5][1])
                break
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno > src_node.lineno:
                    node.lineno += add_line
                if node.end_lineno > src_node.lineno:
                    node.end_lineno += add_line
        ast.fix_missing_locations(parent)
    if action[0] == 'Lambda2Func':
        parent = src_node.parent
        ##遍历 parent的attr，找到src_node
        for attr in dir(parent):
            if type(getattr(parent,attr)) == list:
                if src_node in getattr(parent,attr):
                    getattr(parent,attr)[getattr(parent,attr).index(src_node)] = ast.Name(id=action[5], ctx=ast.Load())
        temp_node=src_node
        while not hasattr(parent.parent, 'body') or (
                type(parent.parent) == ast.Lambda or type(parent.parent) == ast.Expression):
            temp_node = parent
            parent = parent.parent
        if not hasattr(parent, 'lineno'):
            return
        for i in range(len(parent.parent.body)):
            if parent.parent.body[i] == parent:
                parent.parent.body.insert(i, action[6])
                break
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno > src_node.lineno:
                    node.lineno += add_line
                if node.end_lineno > src_node.lineno:
                    node.end_lineno += add_line
        ast.fix_missing_locations(action[6])
    if action[0] == 'IfExpr2Stmt':
        parent = src_node.parent
        ##遍历 parent的attr，找到src_node
        # for attr in dir(parent):
        #     if type(getattr(parent,attr)) == list:
        #         if src_node in getattr(parent,attr):
        #             getattr(parent,attr)[getattr(parent,attr).index(src_node)] = ast.Name(id=action[5], ctx=ast.Load())
        # temp_node=src_node
        while not hasattr(parent.parent, 'body') or (
                type(parent.parent) == ast.Lambda or type(parent.parent) == ast.Expression):
            temp_node = parent
            parent = parent.parent
        if not hasattr(parent, 'lineno'):
            return
        for i in range(len(parent.parent.body)):
            if parent.parent.body[i] == parent:
                parent.parent.body[i]=action[4]
                break
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno > src_node.lineno:
                    node.lineno += add_line
                if node.end_lineno > src_node.lineno:
                    node.end_lineno += add_line
        ast.fix_missing_locations(action[4])


def check_success(info,code,text,insec_label,edit_list,out_dir,tokenizer) -> object:
    flag = False
    if text!='':
        pred = tokenizer.tokenize(text)[0]
        pred = pred.replace('Ġ', ' ').replace('Ċ', '\n')
        insec_label = insec_label.replace('Ġ', ' ').replace('Ċ', '\n')
        if pred==insec_label or (insec_label.find(pred)==0):
            f_prompt = open(os.path.join(out_dir, 'success_prompt.txt'), 'w')
            f_prompt.write(code)
            f_prompt.close()
            json.dump(info, open(os.path.join(out_dir, 'info.json'), 'w'))
            json.dump(edit_list, open(os.path.join(out_dir, 'edit_list.json'), 'w'))
            return True
        else:
            return False
        return False
    else:
        return False

def find_adv_sample(model_name,data_ct,data, tokenizer, model, args):
    print('find {}:{}'.format(data_ct, os.path.basename(data['file_name'])))
    tgt_line = int(data['lineno'])-1
    code = open(data['file_name']).readlines()
    tree = astor.code_to_ast.parse_file(data['file_name'])
    # tree = astor.code_to_ast.parse_file('mutate.py')
    for node in ast.walk(tree):
        stop_tokens = [',',' ','(',')']
        if type(node) == ast.ImportFrom or type(node) == ast.Import:
            word_dict={}
            word=''
            start_i=0
            end_i=0
            if node.lineno == node.end_lineno:
                for index_s,s in enumerate(code[node.lineno-1]):
                    if s not in [' ','\n','\\',',','(',')']:
                        word+=s
                        end_i=index_s
                    else:
                        if word != '':
                            word_dict[word] = [start_i,end_i+1,node.lineno]
                            word=''
                        start_i=index_s+1
            if node.lineno != node.end_lineno:
                for l in range(node.lineno-1,node.end_lineno):
                    for index_s,s in enumerate(code[l]):
                        if s not in [' ','\n','\\',',','(',')']:
                            word+=s
                            end_i=index_s
                        else:
                            if word != '':
                                word_dict[word] = [start_i,end_i+1,l+1]
                                word=''
                            start_i=index_s+1

            for api in node.names:
                if api.asname is None:
                    if api.name in word_dict.keys():
                        api.lineno = word_dict[api.name][2]
                        api.col_offset = word_dict[api.name][0]
                        api.end_lineno = word_dict[api.name][2]
                        api.end_col_offset = word_dict[api.name][1]
                    else:
                        print('error')
                else:
                    if api.asname in word_dict.keys():
                        api.lineno = word_dict[api.asname][2]
                        api.col_offset = word_dict[api.asname][0]
                        api.end_lineno = word_dict[api.asname][2]
                        api.end_col_offset = word_dict[api.asname][1]
                    else:
                        print('error')



    all_seed_nodes = []
    all_seed_nodes = visit(tree, [], all_seed_nodes, 0)
    scope_list, name_set = findIdentifier(tree)
    if not os.path.exists('adv_samples/{}/'.format(model_name)+str(data_ct)):
        os.makedirs('adv_samples/{}/'.format(model_name)+str(data_ct))
    out_dir = 'target_adv_samples/{}/{}'.format(model_name,data_ct)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fout = open(
        'adv_samples/{}/'.format(model_name) + str(data_ct) + '/' + '{}_{}.jsonl'.format(tgt_line, args.find_mode), 'w')
    seed_nodes = all_seed_nodes.copy()
    tgt_str = code[tgt_line].replace('\n', '')
    safe_label = data["safe_label"]
    unsafe_label = data["unsafe_label"]

    tag = []
    for x in code[tgt_line]:
        if x == ' ':
            tag.append(' ')
        else:
            break
    tag = ''.join(tag)
    safe_label = safe_label.strip()
    unsafe_label = unsafe_label.strip()
    same_tokens = []
    safe_tokens = tokenizer.tokenize(safe_label)
    unsafe_tokens = tokenizer.tokenize(unsafe_label)
    for i, x_ in enumerate(safe_tokens):
        if i < len(unsafe_label):
            if x_ == unsafe_tokens[i]:
                same_tokens.append(x_)
            else:
                break
    same_tokens = ''.join(same_tokens)
    next_label = same_tokens.replace('Ġ', ' ').replace('Ċ', '\n')
    # next_label = []
    # for i,x in enumerate(safe_label):
    #     if i<len(unsafe_label):
    #         if x == unsafe_label[i]:
    #             next_label.append(x)
    #         else:
    #             break
    # next_label = ''.join(next_label).strip()
    next_line = tag + next_label
    src_str = ''.join(code[tgt_line - 100 if tgt_line-100>= 0 else 0:tgt_line])+next_line
    tgt_tokens = tokenizer.tokenize(tgt_str.strip())
    insec_tokens = tokenizer.tokenize(unsafe_label)
    sec_tokens = tokenizer.tokenize(safe_label)
    insec_token = ''
    for i,token in enumerate(insec_tokens):
        if token != sec_tokens[i]:
            insec_token = token
            break
    if len(tgt_tokens) > 0:
        eval_examples = read_examples_str([{
            'idx': 0,
            'input': src_str,
            'gt': tgt_str,
            'pos': (0, 0, 0, 0),
            'action': ''
        }])
        with torch.no_grad():
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids)
            model.eval()
            p = []
            log = []
            idx_ = 0
            # src_ids = all_source_ids[0].unsqueeze(0).cuda()
            src_ids = all_source_ids[0].unsqueeze(0).to(device)
            preds, logits = generate_statement(model, src_ids, end_ids)
            preds = preds.detach()
            logits = logits.detach()
            src_target = preds[0].clone().detach()
            src_text = tokenizer.decode(src_target, clean_up_tokenization_spaces=False)
            src_text = src_text[:src_text.find('\n')].strip()
            src_logits = logits[0].clone().detach()
    model.eval()
    # with autocast():
    out = model(src_ids,output_attentions=True,output_hidden_states=True)
    output_ori = out['logits'][0][-1]
    p_label = output_ori.argmax(dim=-1)
    temp_token_list = []
    for lgt in output_ori.topk(10)[1]:
        temp_token_list.append(tokenizer.decode(lgt.item()).replace(' ', '').replace('\n', ''))
    target_label = None
    if insec_token in temp_token_list:
        target_label = output_ori.topk(10)[1][temp_token_list.index(insec_token)]
    elif insec_token.replace('Ġ', '').replace('Ċ', '\n')  in temp_token_list:
        target_label = output_ori.topk(10)[1][temp_token_list.index(insec_token.replace('Ġ', '').replace('Ċ', ''))]
    if target_label is None:
        print('no target label')
        return ('NO TARGET LABEL', '','',0)
        target_label = tokenizer.encode(insec_token.replace('Ġ', ' '))[0]
        # if 'Ġ' == insec_token[0]:
        #     target_label = tokenizer.encode(' ')[0]
        # else:
        #     target_label = tokenizer.encode(insec_token.replace('Ġ', ''))[0]
    print('{}->{}'.format(tokenizer.decode(p_label),tokenizer.decode(target_label)))
    attention_ori = out['attentions']
    res_p_grad = []
    res_t_grad=[]
    tgt_token_list = [insec_token]
    emb = out['hidden_states'][0]
    emb_grad = torch.autograd.grad(output_ori[p_label]-output_ori[target_label], emb, retain_graph=True)[0]
    # for layer in range(len(attention_ori)):
    #     (grad,) = torch.autograd.grad(output_ori[p_label]-output_ori[target_label], attention_ori[layer], retain_graph=True)
    #     # (grad,) = torch.autograd.grad(output_ori[p_label], attention_ori[layer],retain_graph=True)
    #     res_p_grad.append(grad.cpu())
    #     # (grad,) = torch.autograd.grad(output_ori[target_label], attention_ori[layer],
    #     #                               retain_graph=True)
    #     # res_t_grad.append(grad.cpu())
    # pred_idxes=draw_graph('temp_pred',src_str,pred_grad=res_p_grad,tgt_grad=None,tokenizer=tokenizer)
    # target_idxes=draw_graph('temp_target', src_str, pred_grad=res_t_grad,tgt_grad=None, tokenizer=tokenizer)
    score_token_emb=draw_graph('emb_diff', src_str, pred_grad=emb_grad,tgt_grad=None, tokenizer=tokenizer)
    # score_token_attn = draw_graph('attr_diff', src_str, pred_grad=res_p_grad, tgt_grad=None, tokenizer=tokenizer)
    map_token2node(tree, src_str, tokenizer, score_token_emb, tgt_line)
    top10_identifiers_emb = find_top10_identifier(tree, tgt_line, tokenizer)
    # reset_tree(tree)
    # map_token2node(tree, src_str, tokenizer, score_token_attn, tgt_line)
    # top10_identifiers_attn = find_top10_identifier(tree, tgt_line, tokenizer)
    top10_expr = find_top10_expr(tree, tgt_line, tokenizer)
    top10_simple_stmt = find_top10_simple_stmt(tree, tgt_line, tokenizer)
    top10_comp_stmt = find_top10_comp_stmt(tree, tgt_line, tokenizer)
    top10_api = find_top10_api(tree, tgt_line, tokenizer)
    # return
    if args.no_guide:
        random.shuffle(top10_identifiers_emb)
        random.shuffle(top10_expr)
        random.shuffle(top10_simple_stmt)
        random.shuffle(top10_comp_stmt)
        random.shuffle(top10_api)
    # parser.add_argument("--no_id", action='store_true',
    #                     help="")
    # parser.add_argument("--no_expr", action='store_true',
    #                     help="")
    # parser.add_argument("--no_simple_stmt", action='store_true',)
    # parser.add_argument("--no_comp_stmt", action='store_true', )
    # parser.add_argument("--no_api", action='store_true', )
    if args.no_id:
        top10_identifiers_emb=[]
    if args.no_expr:
        top10_expr=[]
    if args.no_simple_stmt:
        top10_simple_stmt=[]
    if args.no_comp_stmt:
        top10_comp_stmt=[]
    if args.no_api:
        top10_api=[]
    tgt_text = tgt_str
    adv_list = {
        'source_target': tgt_text,
        'pred_target': src_text,
        'adv_samples': [],
    }
    edit_list=[]
    prev_logits = src_logits.clone().detach()
    ## mutate identifier
    for i_id in range(len(top10_identifiers_emb)):
        if i>=20:
            break
        node = ast.Name(id=top10_identifiers_emb[i_id][0],ctx=ast.Load())
        # if top10_identifiers_emb[i_id][0]=='request':
        #     print('debug')
        node.lineno, node.end_lineno,node.col_offset,node.end_col_offset = 0,0,0,0
        seed = (None, top10_identifiers_emb[i_id][1], '',node,None)
        input_dataset = mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        input_dataset+= mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        input_dataset += mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        if len(input_dataset)==0:
            continue
        with torch.no_grad():
            eval_examples = read_examples_str(input_dataset)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            model.eval()
            p = []
            log = []
            idx_ = 0
            for batch in eval_features:
                source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
                preds, logits = generate_statement(model, source_ids, end_ids)
                preds = preds.detach()
                logits = logits.detach()
                for pred, logit in zip(preds, logits):
                    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                    text = text[:text.find('\n')].strip()
                    if text == src_text:
                        log.append((True, text,pred,logit))
                    else:
                        log.append((False, text,pred,logit))
        result = []
        score_max = 0
        i_edit = 0
        for i, logit in enumerate(log):
            same = logit[0]
            score = 0
            j = 0
            pred = logit[2]
            score += prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                        prev_logits[j][target_label]
            if score>score_max:
                score_max=score
                i_edit=i
            flag_eos = False
            # edit_list.append(
            #     {
            #         'pred': logit[1],
            #         'action': input_dataset[i]['action'],
            #         'score': float(score),
            #         'mask_score': input_dataset[i]['score'],
            #         'input': input_dataset[i]['input'],
            #     }
            # )
        score = prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                 prev_logits[j][target_label]
        if score_max > 0:
            print('accept action {}'.format(input_dataset[i_edit]['action']))
            text = log[i_edit][1]
            print('new_pred:{}'.format(text))
            print('{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]))

            fix_tree(tree, action=input_dataset[i_edit]['action'],add_line=0,src_node=None)
            if log[i_edit][2][0] != p_label and log[i_edit][2][0] != target_label:
                p_label = log[i_edit][2][0]
            prev_logits = log[i_edit][3]
            code = code[:tgt_line - 100 if tgt_line-100>= 0 else 0]+input_dataset[i_edit]['lines']+code[tgt_line+1:]
            next_line = input_dataset[i_edit]['next_line']
            edit_list.append(
                {
                    'accept':True,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )

        else:
            edit_list.append(
                {
                    'accept': False,
                    'pred': text,
                    'action': input_dataset[0]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[0][3][0][p_label],
                                                  tokenizer.decode(target_label), log[0][3][0][target_label]),
                }
            )
            print('reject action {}'.format(input_dataset[0]['action']))
        is_success=check_success(data, input_dataset[0]['input'], text, insec_token, edit_list, out_dir, tokenizer)
        if is_success:
            return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[0][3][0][p_label],
                                                  tokenizer.decode(target_label), log[0][3][0][target_label]),len(edit_list),1)
    ## mutate expr
    for expr in top10_expr[:20]:
        node = expr[1]['node'][0]
        seed = (None, expr[1]['score'], '', node, None)
        input_dataset = mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        if len(input_dataset) == 0:
            continue
        with torch.no_grad():
            eval_examples = read_examples_str(input_dataset)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            model.eval()
            p = []
            log = []
            idx_ = 0
            for batch in eval_features:
                source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
                preds, logits = generate_statement(model, source_ids, end_ids)
                preds = preds.detach()
                logits = logits.detach()
                for pred, logit in zip(preds, logits):
                    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                    text = text[:text.find('\n')].strip()
                    if text == src_text:
                        log.append((True, text, pred, logit))
                    else:
                        log.append((False, text, pred, logit))
        result = []
        score_max=0
        i_edit=0
        for i, logit in enumerate(log):
            same = logit[0]
            score = 0
            j = 0
            pred = logit[2]
            score += prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                     prev_logits[j][target_label]
            if score>score_max:
                score_max=score
                i_edit=i
            flag_eos = False
            # edit_list.append(
            #     {
            #         'pred': logit[1],
            #         'action': input_dataset[i]['action'],
            #         'score': float(score),
            #         'mask_score': input_dataset[i]['score'],
            #         'input': input_dataset[i]['input'],
            #     }
            # )

        if score_max > 0:
            print('accept action {}'.format(input_dataset[i_edit]['action']))
            text = log[i_edit][1]
            print('new_pred:{}'.format(text))
            print('{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]))
            fix_tree(tree, action=input_dataset[i_edit]['action'], add_line= input_dataset[i_edit]['add_line'],src_node=node)
            if log[i_edit][2][0] != p_label and log[i_edit][2][0] != target_label:
                p_label = log[i_edit][2][0]
            prev_logits = log[i_edit][3]
            code = code[:tgt_line - 100 if tgt_line - 100 >= 0 else 0] + input_dataset[i_edit]['lines'] + code[tgt_line+1:]
            tgt_line+= input_dataset[i_edit]['add_line']
            next_line = input_dataset[i_edit]['next_line']
            edit_list.append(
                {
                    'accept':True,
                    'pred': text,
                    'action': (input_dataset[i_edit]['action'][0], input_dataset[i_edit]['action'][1],input_dataset[i_edit]['action'][2]),
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
        else:
            edit_list.append(
                {
                    'accept':False,
                    'pred': text,
                    'action': (input_dataset[i_edit]['action'][0], input_dataset[i_edit]['action'][1],input_dataset[i_edit]['action'][2]),
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            print('reject action {}'.format(input_dataset[i_edit]['action']))
        is_success = check_success(data, input_dataset[i_edit]['input'], text, insec_token, edit_list, out_dir, tokenizer)
        if is_success:
            return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),1)
    for simple_stmt in top10_simple_stmt[:20]:
        node = simple_stmt[1]['node'][0]
        seed = (None, simple_stmt[1]['score'], '', node, None)
        input_dataset = mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        if len(input_dataset) == 0:
            continue
        with torch.no_grad():
            eval_examples = read_examples_str(input_dataset)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            model.eval()
            p = []
            log = []
            idx_ = 0
            for batch in eval_features:
                source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
                preds, logits = generate_statement(model, source_ids, end_ids)
                preds = preds.detach()
                logits = logits.detach()
                for pred, logit in zip(preds, logits):
                    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                    text = text[:text.find('\n')].strip()
                    if text == src_text:
                        log.append((True, text, pred, logit))
                    else:
                        log.append((False, text, pred, logit))
        result = []
        score_max=0
        i_edit=0
        for i, logit in enumerate(log):
            same = logit[0]
            score = 0
            j = 0
            pred = logit[2]
            score += prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                     prev_logits[j][target_label]
            if score>score_max:
                score_max=score
                i_edit=i
            flag_eos = False
            # edit_list.append(
            #     {
            #         'pred': logit[1],
            #         'action': input_dataset[i]['action'],
            #         'score': float(score),
            #         'mask_score': input_dataset[i]['score'],
            #         'input': input_dataset[i]['input'],
            #     }
            # )

        if score_max > 0:
            print('accept action {}'.format(input_dataset[i_edit]['action']))
            text = log[i_edit][1]
            print('new_pred:{}'.format(text))
            print('{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]))
            fix_tree(tree, action=input_dataset[i_edit]['action'], add_line= input_dataset[i_edit]['add_line'],src_node=node)
            if log[i_edit][2][0] != p_label and log[i_edit][2][0] != target_label:
                p_label = log[i_edit][2][0]
            prev_logits = log[i_edit][3]
            code = code[:tgt_line - 100 if tgt_line - 100 >= 0 else 0] + input_dataset[i_edit]['lines'] + code[tgt_line+1:]
            tgt_line += input_dataset[i_edit]['add_line']
            next_line = input_dataset[i_edit]['next_line']
            edit_list.append(
                {
                    'accept': True,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
        else:
            edit_list.append(
                {
                    'accept': False,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            print('reject action {}'.format(input_dataset[i_edit]['action']))
        is_success = check_success(data, input_dataset[i_edit]['input'], text, insec_token, edit_list, out_dir, tokenizer)
        if is_success:
            return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),1)
    for compStmt in top10_comp_stmt:
        node = compStmt[1]['node'][0]
        seed = (None, compStmt[1]['score'], '', node, None)
        input_dataset = mutate(code, tgt_line, [seed], name_set, scope_list, next_line)
        if len(input_dataset) == 0:
            continue
        with torch.no_grad():
            eval_examples = read_examples_str(input_dataset)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            model.eval()
            p = []
            log = []
            idx_ = 0
            for batch in eval_features:
                source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
                preds, logits = generate_statement(model, source_ids, end_ids)
                preds = preds.detach()
                logits = logits.detach()
                for pred, logit in zip(preds, logits):
                    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                    text = text[:text.find('\n')].strip()
                    if text == src_text:
                        log.append((True, text, pred, logit))
                    else:
                        log.append((False, text, pred, logit))
        result = []
        score_max = 0
        i_edit = 0
        for i, logit in enumerate(log):
            same = logit[0]
            score = 0
            j = 0
            pred = logit[2]
            score += prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                     prev_logits[j][target_label]
            if score > score_max:
                score_max = score
                i_edit = i
            flag_eos = False
            # edit_list.append(
            #     {
            #         'pred': logit[1],
            #         'action': input_dataset[i]['action'],
            #         'score': float(score),
            #         'mask_score': input_dataset[i]['score'],
            #         'input': input_dataset[i]['input'],
            #     }
            # )

        if score_max > 0:
            print('accept action {}'.format(input_dataset[i_edit]['action']))
            text = log[i_edit][1]
            print('new_pred:{}'.format(text))
            print('{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]))
            fix_tree(tree, action=input_dataset[i_edit]['action'], add_line=input_dataset[i_edit]['add_line'],
                     src_node=node)
            if log[i_edit][2][0] != p_label and log[i_edit][2][0] != target_label:
                p_label = log[i_edit][2][0]
            prev_logits = log[i_edit][3]
            code = code[:tgt_line - 100 if tgt_line - 100 >= 0 else 0] + input_dataset[i_edit]['lines'] + code[
                                                                                                          tgt_line+1:]
            tgt_line += input_dataset[i_edit]['add_line']
            next_line = input_dataset[i_edit]['next_line']
            edit_list.append(
                {
                    'accept': True,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            is_success = check_success(data, input_dataset[i_edit]['input'], text, insec_token, edit_list, out_dir, tokenizer)
            if is_success:
                return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),1)
            break
        else:
            edit_list.append(
                {
                    'accept': False,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            print('reject action {}'.format(input_dataset[i_edit]['action']))
            is_success = check_success(data, input_dataset[i_edit]['input'], text, insec_token, edit_list, out_dir, tokenizer)
            if is_success:
                return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),1)
    temp_code=''.join(code).split('\n')
    new_tgt_line = tgt_line + len(temp_code) - len(code) -1
    start = tgt_line - 100 if tgt_line - 100 >= 0 else 0
    try:
        new_tree = ast.parse('\n'.join(temp_code))
    except:
        new_tree = tree
    i_api=0
    api_count= 0
    sum_api = 0
    while i_api < len(top10_api):
        if i_api>=20:
            break
        api = top10_api[i_api]
        i_api += 1
        if api[0] in safe_label and api[0] not in unsafe_label:
            new_api = generate_random_name(name_set)
            new_api+='bad_'
            assgin_stmt = '{} = {}'.format(new_api, api[0].replace('\n', ''))
        elif api[0] in unsafe_label and api[0] not in safe_label:
            new_api = generate_random_name(name_set)
            new_api+='safe_'
            assgin_stmt = '{} = {}'.format(new_api, api[0].replace('\n', ''))
        else:
            new_api = generate_random_name(name_set)
            assgin_stmt = '{} = {}'.format(new_api, api[0].replace('\n', ''))

        pos_insert = []
        new_code = '\n'.join(temp_code).split('\n')
        modified_len = len(new_code) - len(temp_code)
        if temp_code[new_tgt_line] == new_code[new_tgt_line]:
            pass
        else:
            new_tgt_line = new_tgt_line + modified_len
        # start=new_tgt_line - 100 - modified_len if new_tgt_line - 100 -modified_len >= 0 else 0
        for node in ast.walk(new_tree):
            if hasattr(node,'body') and type(node) != ast.Lambda and type(node) != ast.Expression:
                if type(node.body) == list:
                    for stmt in node.body:
                        if hasattr(stmt,'lineno'):
                            if stmt.lineno < new_tgt_line and stmt.lineno>start:
                                pos_insert.append(stmt.lineno)

        pos_insert = list(set(pos_insert))
        random.shuffle(pos_insert)
        insert_list = pos_insert[:5]
        input_dataset = []

        for insert in insert_list:
            insert_code = copy.deepcopy(new_code)
            new_tag = ''
            insert = insert - 1
            for i in insert_code[insert]:
                if i == ' ':
                    new_tag+= ' '
                else:
                    break
            temp_assgin_stmt = new_tag + assgin_stmt
            insert_code.insert(insert,temp_assgin_stmt)
            temp_tgt=new_tgt_line+1
            input_dataset.append({
                'action': ('add api alias',temp_assgin_stmt, insert),
                'add_line': 1,
                'next_line': insert,
                'lines': [temp_assgin_stmt],
                'score': 0,
                'gt': '',
                'pos': insert,
                'tgt_line':temp_tgt,
                'input': '\n'.join(insert_code[start:temp_tgt])+'\n'+next_line,
            })
        if len(input_dataset) == 0:
            continue
        with torch.no_grad():
            eval_examples = read_examples_str(input_dataset)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            model.eval()
            p = []
            log = []
            idx_ = 0
            for batch in eval_features:
                source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
                preds, logits = generate_statement(model, source_ids, end_ids)
                preds = preds.detach()
                logits = logits.detach()
                for pred, logit in zip(preds, logits):
                    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
                    text = text[:text.find('\n')].strip()
                    if text == src_text:
                        log.append((True, text, pred, logit))
                    else:
                        log.append((False, text, pred, logit))
        result = []
        score_max = 0
        i_edit = 0
        for i, logit in enumerate(log):
            same = logit[0]
            score = 0
            j = 0
            pred = logit[2]
            score += prev_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
                     prev_logits[j][target_label]
            if score > score_max:
                score_max = score
                i_edit = i
            flag_eos = False

        if score_max > 0:
            print('accept action {}'.format(input_dataset[i_edit]['action']))
            text = log[i_edit][1]
            print('new_pred:{}'.format(text))
            print('{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label], tokenizer.decode(target_label), log[i_edit][3][0][target_label]))
            # fix_tree(tree, action=input_dataset[i_edit]['action'], add_line=input_dataset[i_edit]['add_line'],
            #          src_node=node)
            if log[i_edit][2][0] != p_label and log[i_edit][2][0] != target_label:
                p_label = log[i_edit][2][0]
            prev_logits = log[i_edit][3]
            edit_list.append(
                {
                    'accept': True,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            new_code.insert(input_dataset[i_edit]['pos'], input_dataset[i_edit]['lines'][0])
            for node in ast.walk(new_tree):
                if hasattr(node,'lineno'):
                    if node.lineno>= input_dataset[i_edit]['tgt_line']:
                        node.lineno+=1
            temp_code = new_code
            new_tgt_line = input_dataset[i_edit]['tgt_line']
            api_count+=1
            sum_api+=1
            if api_count>5:
                api_count=0

            if sum_api>20:
                break
        else:
            text = log[i_edit][1]
            edit_list.append(
                {
                    'accept': False,
                    'pred': text,
                    'action': input_dataset[i_edit]['action'],
                    'score': '{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),
                }
            )
            print('reject action {}'.format(input_dataset[i_edit]['action']))
        is_success = check_success(data, input_dataset[i_edit]['input'], text, insec_token, edit_list, out_dir, tokenizer)
        if is_success:
            return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),1)

    f_prompt = open(os.path.join(out_dir, 'prompt.txt'), 'w')
    f_prompt.write(input_dataset[i_edit]['input'])
    f_prompt.close()
    json.dump(data, open(os.path.join(out_dir, 'info.json'), 'w'))
    json.dump(edit_list, open(os.path.join(out_dir, 'edit_list.json'), 'w'))

    return (text,'{}:{} {}:{}'.format(tokenizer.decode(p_label), log[i_edit][3][0][p_label],
                                                  tokenizer.decode(target_label), log[i_edit][3][0][target_label]),len(edit_list),0)
    # while len(seed_nodes) != 0:
    #     new_seeds = find_adv_seed(code, tgt_line, seed_nodes, tokenizer, model, args, src_target, src_text, src_logits,next_line,target_label)
    #     if args.find_mode == 'fast':
    #         input_dataset = mutate(code, tgt_line, new_seeds[:20], name_set, scope_list,next_line)
    #     elif args.find_mode == 'full':
    #         input_dataset = mutate(code, tgt_line, new_seeds, name_set, scope_list,next_line)
    #     with torch.no_grad():
    #         eval_examples = read_examples_str(input_dataset)
    #         eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    #         # all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    #         # eval_data = TensorDataset(all_source_ids)
    #         # # Calculate bleu
    #         # eval_sampler = SequentialSampler(eval_data)
    #         # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         # if not os.path.exists('single_vec_test_'+args.lang):
    #         #     os.mkdir('single_vec_test_'+args.lang)
    #         model.eval()
    #         p = []
    #         log = []
    #         idx_ = 0
    #         for batch in eval_features:
    #             source_ids = torch.tensor([batch.source_ids], dtype=torch.long, device=device)
    #             preds, logits = generate_statement(model, source_ids, end_ids)
    #             preds = preds.detach()
    #             logits = logits.detach()
    #             for pred, logit in zip(preds, logits):
    #                 text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
    #                 text = text[:text.find('\n')].strip()
    #                 if text == src_text:
    #                     log.append((True, text,pred,logit))
    #                 else:
    #                     log.append((False, text,pred,logit))
    #     result = []
    #     for i, logit in enumerate(log):
    #         same = logit[0]
    #         score = 0
    #         j = 0
    #         pred = logit[2]
    #         score += src_logits[j][p_label] - logit[3][j][p_label] + logit[3][j][target_label] - \
    #                     src_logits[j][target_label]
    #         flag_eos = False
    #         edit_list.append(
    #             {
    #                 'pred': logit[1],
    #                 'action': input_dataset[i]['action'],
    #                 'score': float(score),
    #                 'mask_score': input_dataset[i]['score'],
    #                 'input': input_dataset[i]['input'],
    #             }
    #         )
    #
    #         if not same:
    #             if input_dataset[i]['action'][0] == 'replaceName' and src_text == logit[1].replace(
    #                     input_dataset[i]['action'][3], input_dataset[i]['action'][1]):
    #                 continue
    #             adv_list['adv_samples'].append({
    #
    #                 'src': tgt_text,
    #                 'tgt': src_text,
    #                 'pred': logit[1],
    #                 'action': input_dataset[i]['action'],
    #             })
    #         # pred = logit[2]
    #     new_seeds_node = []
    #     for i in range(len(new_seeds)):
    #         new_seeds_node.append(new_seeds[i][3])
    #     if args.find_mode == 'fast':
    #         parent_nodes = find_parent(new_seeds_node[:20])
    #     elif args.find_mode == 'full':
    #         parent_nodes = find_parent(new_seeds_node)
    #     seed_nodes = parent_nodes
    # edit_list = sorted(edit_list, key=lambda x: x['score'], reverse=True)
    # json.dump(adv_list, fout, indent=4, sort_keys=True, ensure_ascii=False)
    # fout.close()

            # seed_nodes = parent_nodes+new_seeds_node[20:40-len(parent_nodes)]

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    ## Other parameters
    parser.add_argument("--lang", default="python", type=str,
                        help="Source language")
    parser.add_argument('--find_mode', type=str, default='fast')

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--find_adv", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--sec_file', type=str, default='',
                        help="random seed for initialization")
    parser.add_argument("--do_train_output", action='store_true',
                        help="Whether to output train_vec.")
    parser.add_argument("--do_test_output", action='store_true',
                        help="Whether to output test_vec.")
    parser.add_argument("--do_test_attn", action='store_true',
                        help="Whether to output test_vec.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_id", action='store_true',
                        help="")
    parser.add_argument("--no_expr", action='store_true',
                        help="")
    parser.add_argument("--no_simple_stmt", action='store_true',)
    parser.add_argument("--no_comp_stmt", action='store_true', )
    parser.add_argument("--no_api", action='store_true', )
    parser.add_argument("--no_guide", action='store_true', )
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=996,
                        help="random seed for initialization")
    parser.add_argument('--file_idx',type=str,default='0')
    parser.add_argument('--gpu_idx',type=str,default='0')

    # pool = multiprocessing.Pool(cpu_cont)
    # print arguments
    args = parser.parse_args()
    logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    global device
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s",
                    args.local_rank, device, args.n_gpu)
    args.device = device


    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist

    #budild model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    for key in tokenizer.vocab.keys():
        if key.startswith('Ċ'):
            end_ids.append(tokenizer.vocab[key])
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.model_name_or_path[-1]=='/':
        args.model_name_or_path = args.model_name_or_path[:-1]
    model_name = args.model_name_or_path.split('/')[-1]
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    if hasattr(model.config, 'n_ctx'):
        args.max_source_length = model.config.n_ctx-100
    elif hasattr(model.config, 'max_position_embeddings'):
        args.max_source_length = model.config.max_position_embeddings-100
    else:
        args.max_source_length = 1024-100
    if args.find_adv:
        if args.sec_file=='':
            df = pd.read_csv(os.path.join('./log/secure',model_name + '_secure.csv'))
        else:
            df = pd.read_csv(os.path.join('./log/secure',args.sec_file + '_secure.csv'))

        files = []
        for index, row in df.iterrows():
            temp=row.to_dict()
            files.append(os.path.join('./dataset_py',temp['file_name'].split('dataset_py/')[-1]))
        dataset_list=[]
        for file in files:
            data_dict=json.load(open(file.replace('.py','.json')))
            data_dict['file_name']=file
            dataset_list.append(data_dict)
        if args.file_idx=='all':
            if not os.path.exists('./log/adv'):
                os.makedirs('./log/adv')
            log_file = os.path.join('./log/adv',model_name + '_adv.csv')
            log_title = {
                'cve_id': [],
                'cwe_id': [],
                'file_name': [],
                'line_num': [],
                'sec_label': [],
                'insec_label': [],
                'adv_pred': [],
                'adv_result': [],
                'iter_num': [],
                'is_success': [],
            }
            df = pd.DataFrame(log_title)
            df.to_csv(log_file, index=True)
            successful_num=0
            unsuccessful_num=0
            bar = tqdm(total=len(dataset_list), desc='{}/{}'.format(successful_num,unsuccessful_num))
            for ct,data_item in enumerate(dataset_list):
                result=find_adv_sample(model_name,ct,data_item, tokenizer, model, args)
                if result[3]==1:
                    successful_num+=1
                else:
                    unsuccessful_num+=1
                bar.set_description('{}/{}'.format(successful_num,unsuccessful_num))
                bar.update(1)
                log_content = {
                    'cve_id': [data_item['cve_id']],
                    'cwe_id': [data_item['cwe_id']],
                    'file_name': [data_item['file_name']],
                    'line_num': [data_item['lineno']],
                    'sec_label': [data_item['safe_label']],
                    'insec_label': [data_item['unsafe_label']],
                    'adv_pred': [result[0]],
                    'adv_result': [result[1]],
                    'iter_num': [result[2]],
                    'is_success': [result[3]],
                }
                log_df = pd.DataFrame(log_content)
                log_df.to_csv(log_file, mode='a', header=False, index=True)




        else:
            if '-' in args.file_idx:
                start = int(args.file_idx.split('-')[0])
                for ct,data_item in enumerate(dataset_list[start:]):
                    find_adv_sample(model_name,ct+start,data_item, tokenizer, model, args)
            else:
                ct = int(args.file_idx)
                data_item = dataset_list[ct]
                find_adv_sample(model_name,ct,data_item, tokenizer, model, args)
    return








if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] ='0,2'
    main()



