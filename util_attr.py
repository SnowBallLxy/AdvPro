import os.path

import numpy
import numpy as np
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from tqdm import tqdm

import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)
def tokenize(item):
    source, max_length, tokenizer = item
    source_tokens = [x for x in tokenizer.tokenize(source) if x!='\u0120']
    source_tokens = ["<s>","<decoder-only>","</s>"]+source_tokens[-(max_length-3):]
    return source_tokens


token_gap=0.1
word_width = 0.2
space_width = 0.2
def draw_linetext(texts, position, labels,startx,starty, size=15, color='black'):
    x = startx
    y = starty
    token_gap=0.1
    word_width = 0.2
    space_width = 0.2
    for text in texts:
        x += token_gap+(word_width*len(labels[text])/2)
        position[text] = (x,y)
        x += word_width*len(labels[text])/2
def draw_code(codes,fix_position,labels):
    out=[]
    # codes = codes[3:]
    starty= 45
    line_gap= 0.5
    for token in codes:
        out.append(token)
        if "\n" in labels[token]:
            draw_linetext(out, fix_position,labels,0, starty, size=15, color='black')
            out = []
            starty-=line_gap
    draw_linetext(out, fix_position,labels,0, starty, size=15, color='black')

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 pred,
                 correct
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.pred = pred
        self.correct = correct


def read_examples(filename):
    """Read examples from filename."""
    examples = []

    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if ".txt" in filename:
                inputs = line.strip().replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = []
            elif 'long' in filename:
                js = json.loads(line)
                inputs = js["input"].replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = js["target"]
                if outputs in ['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
                               'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
                               'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']:
                    continue
                if 'id' in js:
                    idx = js['id']
                pred = js['pred']
                correct = js['correct']

            else:
                js = json.loads(line)
                inputs = js["input"].replace("<EOL>", "</s>").split()
                inputs = inputs[1:]
                inputs = " ".join(inputs)
                outputs = js["gt"]
                if 'id' in js:
                    idx = js['id']
            examples.append(
                Example(
                    idx=idx,
                    source=inputs,
                    target=outputs,
                    pred=pred,
                    correct=correct
                )
            )
    return examples

def draw_graph(filename,tokens,pred_grad=None,tgt_grad=None,tokenizer=None):
    if type(pred_grad) is not list:
        score_sorted = draw_graph_emb(filename,tokens,pred_grad,tgt_grad,tokenizer)
    else:
        score_sorted = draw_graph_att(filename,tokens,pred_grad,tgt_grad,tokenizer)
    return score_sorted

def draw_graph_emb(filename,tokens,pred_grad=None,tgt_grad=None,tokenizer=None):
    tokens = tokenizer.tokenize(tokens)
    labels = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in tokens]
    attrs = pred_grad.norm(dim=-1).detach().cpu().numpy()[0]
    min_attributions = attrs.min()
    max_attributions = attrs.max()
    seq_length = len(tokens)
    normalized_attributions = np.sqrt((attrs - min_attributions) / (max_attributions - min_attributions))
    threshold = 0
    fixed_list = [-1 for i in range(seq_length)]
    edges = []

    # find the top node
    size_list = [float(0) for i in range(seq_length)]
    score_arg = numpy.zeros(seq_length)
    for i in range(seq_length):
        if normalized_attributions[i]> threshold:
            size_list[i] = normalized_attributions[i]
            score_arg[i] = normalized_attributions[i]
    # 执行min-max缩放，将归因分数归一化到[0, 1]范围内
    sort_arg = np.argsort(score_arg)[::-1]
    fig1 = plt.figure(1, figsize=(20, 50))
    fig1.patch.set_facecolor('xkcd:white')

    cmap = plt.cm.get_cmap('rainbow')
    G = nx.DiGraph()
    # draw_code(tokens)
    for i in range(len(tokens)):
        tokens[i] = tokens[i] + str(i)
    for token in tokens:
        G.add_node(token)

    for (i_token, j_token) in edges:
        G.add_weighted_edges_from([(tokens[i_token], tokens[j_token], 0.5)])

    fix_position = {tokens[i]: [i % 20, 50 - i // 20] for i in range(len(tokens))}
    fix_labels = {tokens[i]: labels[i] for i in range(len(tokens))}

    M = G.number_of_edges()
    pos = nx.spring_layout(G, pos=fix_position)
    edge_colors = range(2, M + 2)

    unused_node = list(nx.isolates(G))
    # for node in unused_node:
    #     G.remove_node(node)

    edge_alphas = []
    edges_list = list(G.edges.data())
    for single_edge in edges_list:
        edge_alphas.append(single_edge[2]["weight"])
    cmap = plt.get_cmap('Reds')
    size_ = []
    colors = []
    label_colors = []
    cmap_label = plt.get_cmap('Greys')
    use_pos = {}
    unused_pos = {}
    use_label = {}
    unused_label = {}
    draw_code(tokens, fix_position, fix_labels)
    shown_node = []
    # for node in unused_node:
    #     if fix_position[node][1] < fix_position[list(G.nodes.keys())[0]][1]+2:
    #         shown_node.append(node)

    # fix_position[tokens[top_token_index]] = (
    #     fix_position[tokens[top_token_index]][0], fix_position[tokens[top_token_index]][1] - 1)
    for i in range(len(tokens)):
        # if tokens[i] in unused_node:
        #     # if tokens[i] in shown_node:
        #     label_colors.append(cmap_label(0.5))
        #     unused_pos[tokens[i]] = fix_position[tokens[i]]
        #     unused_label[tokens[i]] = fix_labels[tokens[i]]
        # else:
        size_.append(size_list[i] * 700)
        col = cmap(size_list[i] * 0.9)
        colors.append(col)
        label_colors.append(cmap_label(0.99))
        use_pos[tokens[i]] = fix_position[tokens[i]]
        use_label[tokens[i]] = fix_labels[tokens[i]]
    nx.draw
    # if example.target != '':
    #     target_x = fix_position[tokens[top_token_index]][0] + space_width + (
    #             word_width * len(tokens[top_token_index]) / 2) + (word_width * len(example.target) / 2)
    #     pred_x = target_x = fix_position[tokens[top_token_index]][0] + space_width + (
    #             word_width * len(tokens[top_token_index]) / 2) + (word_width * len(example.pred) / 2)
    #     use_pos[example.target] = (
    #         target_x, fix_position[tokens[top_token_index]][1])
    #     use_label[example.target] = example.target
    #     pred_pos = (pred_x, fix_position[tokens[top_token_index]][1] - 0.5)
    #     pred_label = example.pred
    G.add_node('<_final_>')
    fix_position['<_final_>'] = (20, 0.5)
    size_.append(0.0 * 700)
    col = cmap(0 * 0.9)
    colors.append(col)
    use_pos['<_final_>'] = (20, 0.5)
    use_label['<_final_>'] = '_'

    nodes = nx.draw_networkx_nodes(G, fix_position, node_size=size_, node_color=colors, )
    # edges = nx.draw_networkx_edges(G, fix_position, node_size=size_, arrowstyle='->',
    #                                arrowsize=10, edge_color='b',
    #                                edge_cmap=plt.cm.Blues, width=2)
    # nx.draw_networkx_labels(G, fix_position, labels=fix_labels,font_size=15, font_family='sans-serif')
    # nx.draw_networkx_labels(G, use_pos, labels=use_label, font_size=15, font_family='sans-serif',
    #                         bbox=dict(boxstyle="square,pad=0.0 1", fc="none", ec="k", lw=0.72))
    # nx.draw_networkx_labels(G, unused_pos, labels=unused_label, font_color='gray', font_size=15,
    #                         font_family='sans-serif')

    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # pc.set_array(edge_colors)

    # ax = plt.gca()
    # ax.set_axis_off()
    # # plt.show()
    # f = plt.gcf()
    # output_path = 'pic/' + filename
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # f.savefig(os.path.join(output_path, "{0}.png".format('temp')), bbox_inches='tight', pad_inches=0)
    # f.clear()
    return score_arg

def draw_graph_att(filename,tokens,pred_grad=None,tgt_grad=None,tokenizer=None):
    tokens = tokenizer.tokenize(tokens)
    labels = [token.replace('Ġ', ' ').replace('Ċ','\n') for token in tokens]
    # if not os.path.exists(filename):
    #     return
    att_all = torch.stack(pred_grad)
    att_all = att_all.sum(1).numpy()
    att_all = att_all.sum(1)
    layer_num = att_all.shape[0]
    proportion_all = copy.deepcopy(att_all)
    proportion_all = np.mean(proportion_all, axis=0)
    proportion_all = abs(proportion_all)
    # proportion_all[-1] = np.log10(np.multiply(proportion_all[-1],proportion_all.sum(1)))
    # proportion_all *= proportion_all.sum(1)
    if tgt_grad is not None:
        att_all = torch.stack(tgt_grad)
        att_all = att_all.sum(1).numpy()
        att_all = att_all.sum(1)
        layer_num = att_all.shape[0]
        proportion_all_t = copy.deepcopy(att_all)
        proportion_all_t = np.max(proportion_all_t, axis=0)
        # proportion_all_t *= proportion_all_t.sum(1)
        proportion_all = proportion_all-proportion_all_t
    # proportion_all = abs(proportion_all)
    proportion_all /= abs(proportion_all[1:, :].max())

    # adjust the threshold
    threshold = 0.0
    proportion_all *= (proportion_all > threshold).astype(int)

    seq_length = len(proportion_all[0])
    height_list = [0 for i in range(seq_length)]
    size_list = [float(1) for i in range(seq_length)]
    # -1: not appear  0: appear but not fixed  1: fixed
    fixed_list = [-1 for i in range(seq_length)]
    edges = []

    # find the top node
    ig_remain = [0 for i in range(seq_length)]
    att_combine_layer = att_all.sum(0) / abs(att_all.sum(0).max())
    att_combine_layer *= (1 - np.identity(len(att_combine_layer))) * (att_combine_layer > 0)
    att_combine_layer[0] *= 0
    arg_res = np.argsort(att_combine_layer.sum(-1))[::-1]

    top_token_index = seq_length - 1
    height_list[top_token_index] = 11 / 12

    fixed_list[top_token_index] = 0
    score_arg= numpy.zeros(seq_length)
    for i in range(seq_length):
        if i != top_token_index and proportion_all[top_token_index][i] > threshold:
            fixed_list[i] = 0
            fixed_list[top_token_index] = 1
            size_list[i] = proportion_all[top_token_index][i]
            score_arg[i] = proportion_all[top_token_index][i]
            edges.append((top_token_index,i))

    # for i_token in range(1, seq_length):
    #     for j_token in range(0, seq_length):
    #         if proportion_all[i_token][j_token] < threshold or fixed_list[i_token] == -1:
    #             continue
    #         if fixed_list[j_token] == 1:
    #             pass
    #         if (i_token, j_token) in edges:
    #             continue
    #         if fixed_list[i_token] == 0 and fixed_list[j_token] == 0:
    #             pass
    #             # continue
    #         if fixed_list[i_token] == 1 and fixed_list[j_token] == 0:
    #             pass
    #             # continue
    #         if fixed_list[i_token] == 0 and fixed_list[j_token] == -1:
    #             fixed_list[i_token] = 1
    #             fixed_list[j_token] = 0
    #             height_list[j_token] = ((height_list[i_token]) * 12 - 1) / 12
    #             size_list[j_token] = size_list[i_token] * proportion_all[i_token][j_token]
    #         if fixed_list[i_token] == 1 and fixed_list[j_token] == -1:
    #             fixed_list[j_token] = 0
    #             height_list[j_token] = min(height_list)
    #             size_list[j_token] = size_list[i_token] * proportion_all[i_token][j_token]
    #         score_arg[j_token] = size_list[j_token]
    #         edges.append((i_token, j_token))
    sort_arg = np.argsort(score_arg)[::-1]
    # token examples
    # tokens = ["[CLS]", "i", "don", "'", "t", "know", "um", "do", "you", "do", "a", "lot", "of", "camping", "[SEP]", "I", "know", "exactly", ".", "[SEP]"]
    # tokens = ["[CLS]", "The", "new", "rights", "are", "nice", "enough", "[SEP]", "Everyone", "really", "likes", "the", "newest", "benefits", "[SEP]"]
    # tokens = ["[CLS]", "so", "i", "have", "to", "find", "a", "way", "to", "supplement", "that", "[SEP]", "I", "need", "a", "way", "to", "add", "something", "extra", ".", "[SEP]"]

    fig1 = plt.figure(1, figsize=(20, 50))
    fig1.patch.set_facecolor('xkcd:white')

    cmap = plt.cm.get_cmap('rainbow')
    G = nx.DiGraph()
    # draw_code(tokens)
    for i in range(len(tokens)):
        tokens[i] = tokens[i] + str(i)
    for token in tokens:
        G.add_node(token)

    for (i_token, j_token) in edges:
        G.add_weighted_edges_from([(tokens[i_token], tokens[j_token], 0.5)])

    fix_position = {tokens[i]: [i / len(tokens), height_list[i]] for i in range(len(tokens))}
    fix_position = {tokens[i]: [i % 20, 50 - i // 20] for i in range(len(tokens))}
    fix_labels = {tokens[i]: labels[i] for i in range(len(tokens))}

    M = G.number_of_edges()
    pos = nx.spring_layout(G, pos=fix_position)
    edge_colors = range(2, M + 2)

    unused_node = list(nx.isolates(G))
    for node in unused_node:
        G.remove_node(node)

    edge_alphas = []
    edges_list = list(G.edges.data())
    for single_edge in edges_list:
        edge_alphas.append(single_edge[2]["weight"])
    cmap = plt.get_cmap('Reds')
    size_ = []
    colors = []
    label_colors = []
    cmap_label = plt.get_cmap('Greys')
    use_pos = {}
    unused_pos = {}
    use_label = {}
    unused_label = {}
    draw_code(tokens, fix_position, fix_labels)
    shown_node = []
    # for node in unused_node:
    #     if fix_position[node][1] < fix_position[list(G.nodes.keys())[0]][1]+2:
    #         shown_node.append(node)

    fix_position[tokens[top_token_index]] = (
        fix_position[tokens[top_token_index]][0], fix_position[tokens[top_token_index]][1] - 1)
    for i in range(len(tokens)):
        if tokens[i] in unused_node:
            # if tokens[i] in shown_node:
            label_colors.append(cmap_label(0.5))
            unused_pos[tokens[i]] = fix_position[tokens[i]]
            unused_label[tokens[i]] = fix_labels[tokens[i]]
        else:
            size_.append(size_list[i] * 700)
            col = cmap(size_list[i] * 0.9)
            colors.append(col)
            label_colors.append(cmap_label(0.99))
            use_pos[tokens[i]] = fix_position[tokens[i]]
            use_label[tokens[i]] = fix_labels[tokens[i]]
    nx.draw
    # if example.target != '':
    #     target_x = fix_position[tokens[top_token_index]][0] + space_width + (
    #             word_width * len(tokens[top_token_index]) / 2) + (word_width * len(example.target) / 2)
    #     pred_x = target_x = fix_position[tokens[top_token_index]][0] + space_width + (
    #             word_width * len(tokens[top_token_index]) / 2) + (word_width * len(example.pred) / 2)
    #     use_pos[example.target] = (
    #         target_x, fix_position[tokens[top_token_index]][1])
    #     use_label[example.target] = example.target
    #     pred_pos = (pred_x, fix_position[tokens[top_token_index]][1] - 0.5)
    #     pred_label = example.pred
    G.add_node('<_final_>')
    fix_position['<_final_>'] = (20, 0.5)
    size_.append(0.0 * 700)
    col = cmap(0 * 0.9)
    colors.append(col)
    use_pos['<_final_>'] = (20, 0.5)
    use_label['<_final_>'] = '_'

    nodes = nx.draw_networkx_nodes(G, fix_position, node_size=size_, node_color=colors, )
    # edges = nx.draw_networkx_edges(G, fix_position, node_size=size_, arrowstyle='->',
    #                                arrowsize=10, edge_color='b',
    #                                edge_cmap=plt.cm.Blues, width=2)
    # nx.draw_networkx_labels(G, fix_position, labels=fix_labels,font_size=15, font_family='sans-serif')
    nx.draw_networkx_labels(G, use_pos, labels=use_label, font_size=15, font_family='sans-serif',
                            bbox=dict(boxstyle="square,pad=0.0 1", fc="none", ec="k", lw=0.72))
    nx.draw_networkx_labels(G, unused_pos, labels=unused_label, font_color='gray', font_size=15,
                            font_family='sans-serif')

    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    # plt.show()
    f = plt.gcf()
    output_path = 'pic/' + filename
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f.savefig(os.path.join(output_path, "{0}.png".format('temp')), bbox_inches='tight', pad_inches=0)
    f.clear()
    return score_arg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--attr_file",
                        default=None,
                        nargs='+',
                        type=str,
                        help="The file of attribution scores.")
    parser.add_argument("--tokens_file",
                        default=None,
                        nargs='+',
                        type=str,
                        help="The file that contains tokens of the target example.")

    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)



if __name__ == "__main__":
    main()