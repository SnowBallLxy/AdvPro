import ast
import copy
import random

import astor
import sys
import os
import argparse
import re
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from io import BytesIO
import json
import string
from joblib import Parallel, delayed
from tqdm import tqdm 

random.seed(2023)

lits = json.load(open("literals.json"))

def visit(node, nodes, pindex):
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
    index = len(nodes)
    nodes.append(index)
    for n in ast.iter_child_nodes(node):
        n.parent = node
        visit(n, nodes, index)


filename = sys.argv[0]
if len(sys.argv) > 1:
    filename = sys.argv[1]

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


def process_statement(nodes):

    return None

def find_parent(seed_nodes):
    i = 0
    while i < len(seed_nodes):
        node = seed_nodes[i]
        if hasattr(node, 'parent'):
            if type(node) != ast.FunctionDef and type(node) != ast.Module:
                if node.parent not in seed_nodes and hasattr(node.parent, 'lineno'):
                    seed_nodes.append(node.parent)
            i += 1
            continue
        i += 1
    return None
def findCondition(nodes):
    condition_nodes = []
    for node in nodes:
        if type(node)== ast.Compare:
            condition_nodes.append(node)
            src = astor.to_source(node)
            print(src)
    return condition_nodes

def get_scope(node):
    scope_name = node.name
    while node is not None and hasattr(node, 'parent'):
        if type(node) == ast.FunctionDef or type(node) == ast.ClassDef:
            scope_name = node.name + '.' + scope_name
        node = node.parent
    return scope_name

def findIdentifier(tree):
    name_set = set()
    scope_list={'#root':{'##scope##':(0, 0)}}
    var_locs = scope_list['#root']
    stack = [tree]
    scope_name = '#root'
    current_func_name = '#root'
    current_parent_node = None
    while stack:
        curr_node = stack.pop()
        if hasattr(curr_node, 'parent'):
            if curr_node.parent == current_parent_node and scope_name!='#root':
                scope_name = scope_name[:scope_name.rfind('.')]
                if scope_name not in scope_list:
                    scope_list[scope_name] = {'##scope##':(curr_node.lineno, curr_node.end_lineno)}
        if isinstance(curr_node, ast.FunctionDef) or isinstance(curr_node, ast.ClassDef):
            scope_name = scope_name+ '.' + curr_node.name
            if scope_name not in scope_list:
                scope_list[scope_name] = {'##scope##':(curr_node.lineno, curr_node.end_lineno)}
            current_parent_node = curr_node.parent
        for _, value in ast.iter_fields(curr_node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        stack.append(item)
            elif isinstance(value, ast.AST):
                stack.append(value)
        node = curr_node
        if isinstance(curr_node, ast.Name):
            name_set.add(curr_node.id)
            var_locs = scope_list[scope_name]
            var_name = node.id
            # if var_name == "input_feed":
            #     print()
            if var_name not in var_locs:
                var_locs[var_name] = {
                    "assign": [],
                    "use": []
                }
            if isinstance(node.ctx, ast.Store):
                var_locs[var_name]["assign"].append((node.lineno, node.col_offset))
            elif isinstance(node.ctx, ast.Load):
                var_locs[var_name]["use"].append((node.lineno, node.col_offset))
    return scope_list, name_set

def generate_random_name(name_set=set()):
    # 生成随机长度
    length = random.randint(3, 10)

    # 生成随机字母序列
    letters = string.ascii_lowercase
    while True:
        function_name = ''.join(random.choice(letters) for _ in range(length))
        if function_name not in name_set:
            name_set.add(function_name)
            break

    return function_name

# def mutate_var(scope_list):
#     for scope_name, var_locs in scope_list.items():
#         for var_name, locs in var_locs.items():
#             if len(locs['assign']) > 0:

def findIfExpr(tree):
    if_expr_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.IfExp:
            if_expr_nodes.append(node)
    return if_expr_nodes

def mutateIfExpr2Stmt(nodes):
    action_list=[]
    for node in nodes:
        def create_if_stmt(test, body, orelse):
            return ast.If(test=test, body=body, orelse=orelse)

        # Helper function to replace an IfExp node with an If node
        def replace_ifexpr_with_if(node):
            # Create the new If node
            if_stmt = create_if_stmt(node.test, [ast.Assign(targets=node.parent.targets, value=node.body)], [ast.Assign(targets=node.parent.targets, value=node.orelse)])

            # Replace the IfExp node with the new If node
            ast.copy_location(if_stmt, node)
            ast.fix_missing_locations(if_stmt)
            return if_stmt

        # Walk the syntax tree and replace IfExp nodes with If nodes
        if isinstance(node, ast.IfExp):
            if hasattr(node.parent, 'targets'):
                if_stmt = replace_ifexpr_with_if(node)
                if_str = astor.to_source(if_stmt)
                if node.parent.lineno == node.lineno and node.lineno == node.end_lineno:
                    action_list.append(('IfExpr2Stmt', 'all',[node.parent.lineno], if_str,if_stmt))
    return action_list

def notConditon(test):
    orig_test = test
    condition_list = []
    condition_list.append(ast.UnaryOp(op=ast.Not(), operand=test))
    if hasattr(test, 'op'):
        test = copy.copy(test)
        if type(test.op)==ast.Eq:
            test.op = ast.NotEq()
        elif type(test.op)==ast.NotEq:
            test.op = ast.Eq()
        elif type(test.op)==ast.Lt:
            test.op = ast.GtE()
        elif type(test.op)==ast.LtE:
            test.op = ast.Gt()
        elif type(test.op)==ast.Gt:
            test.op = ast.LtE()
        elif type(test.op)==ast.GtE:
            test.op = ast.Lt()
        elif type(test.op)==ast.Is:
            test.op = ast.IsNot()
        elif type(test.op)==ast.IsNot:
            test.op = ast.Is()
        elif type(test.op)==ast.In:
            test.op = ast.NotIn()
        elif type(test.op)==ast.NotIn:
            test.op = ast.In()
        elif type(test.op)==ast.Not:
            test = test.operand
        condition_list.append(test)
    if hasattr(orig_test,'ops') and type(orig_test) == ast.Compare:
        test = copy.copy(orig_test)
        if len(test.ops)==1:
            if type(test.ops[0]) == ast.Eq:
                test.ops = [ast.NotEq()]
            elif type(test.ops[0]) == ast.NotEq:
                test.ops = [ast.Eq()]
            elif type(test.ops[0]) == ast.Lt:
                test.ops =[ast.GtE()]
            elif type(test.ops[0]) == ast.LtE:
                test.ops =[ast.Gt()]
            elif type(test.ops[0]) == ast.Gt:
                test.ops =[ast.LtE()]
            elif type(test.ops[0]) == ast.GtE:
                test.ops =[ast.Lt()]
            elif type(test.ops[0]) == ast.Is:
                test.ops =[ast.IsNot()]
            elif type(test.ops[0]) == ast.IsNot:
                test.ops =[ast.Is()]
            elif type(test.ops[0]) == ast.In:
                test.ops =[ast.NotIn()]
            elif type(test.ops[0]) == ast.NotIn:
                test.ops =[ast.In()]
            condition_list.append(test)
    return condition_list

def negate_if_expr(nodes):
    """
    将 if 表达式的 test 取反，同时调换 body 和 orelse 的位置

    :param if_expr: ast.If 表达式
    :return: 修改后的 ast.If 表达式
    """
    action_list = []
    for if_expr in nodes:
    # 取反 test
        if not hasattr(if_expr.parent, 'targets'):
            continue
        new_test_list = notConditon(if_expr.test)

        # 调换 body 和 orelse
        for test in new_test_list:
            new_test = test
            new_body = if_expr.orelse
            new_orelse = if_expr.body

            # 生成修改后的 if 表达式
            new_if_expr = ast.IfExp(test=new_test, body=new_body, orelse=new_orelse)
            new_if_str = astor.to_source(ast.Assign(targets = if_expr.parent.targets, value=new_if_expr))
            if if_expr.parent.lineno == if_expr.lineno:
                action_list.append(('negate_if_expr','all',[if_expr.parent.lineno,if_expr.parent.end_lineno], new_if_str))



    return action_list

def findAssgin(tree):
    assign_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.Assign:
            if type(node.value) == ast.List or type(node.value) == ast.Tuple or type(node.value) == ast.Set or type(node.value) == ast.Dict or type(node.value) == ast.Call:
                assign_nodes.append(node)
    # if len(assign_nodes) >100:
    #     assign_nodes = random.sample(assign_nodes, 100)
    return assign_nodes

def mutateAssginAddLine(nodes):
    action_list = []
    for node in nodes:
        flag=0
        for n in ast.walk(node):
            if type(n) == ast.Constant and type(n.value) == str:
                if n.value.find('{') != -1 or n.value.find('}') != -1 or n.value.find('[') != -1 or n.value.find(']') != -1 or n.value.find('(') != -1 or n.value.find(')') != -1:
                    flag=1
        if node.lineno == node.end_lineno and flag==0:
            new_node_str = astor.to_source(node)
            action_list.append(('AssginAddLine','all',[node.lineno], new_node_str))
    return action_list
def findLambda(tree):
    lambda_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.Lambda:
            lambda_nodes.append(node)
    return lambda_nodes



def mutateLambda2func(nodes, name_set=set()):
    action_list = []
    for lambda_expr in nodes:
        args = lambda_expr.args
        body = lambda_expr.body
        name=generate_random_name(name_set)
        # 生成函数定义节点
        func_def = ast.FunctionDef(
            name=name,  # 使用特殊名称 '<lambda>'
            args=args,
            body=[ast.Return(value=body)],  # 将主体内容放到函数体中并使用 Return 语句返回
            decorator_list=[],
            returns=None
        )
        func_str = astor.to_source(func_def)
        parent = lambda_expr.parent
        while not hasattr(parent.parent, 'body') or (type(parent.parent)==ast.Lambda or type(parent.parent)==ast.Expression):
            parent = parent.parent
        if not hasattr(parent,'lineno'):
            continue
        action_list.append(('Lambda2Func','all',[parent.lineno], func_str,(lambda_expr.lineno,lambda_expr.end_lineno,lambda_expr.col_offset,lambda_expr.end_col_offset),name,(func_def)))
    return action_list



def findComp(tree):
    comp_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.ListComp or type(node) == ast.SetComp or type(node) == ast.DictComp:
            comp_nodes.append(node)
    return comp_nodes


def mutateComp2For(nodes):
    action_list = []
    for node in nodes:
        if hasattr(node.parent, 'targets'):
            generators = node.generators
            target = node.parent.targets[0]
            # 构造等价的完整语句
            if type(node) == ast.ListComp:
                elt = node.elt
                temp = ast.Assign(targets=[target], value=ast.List(elts=[], ctx=ast.Load()))
                loop_body = [ast.Expr(
                    value=ast.Call(func=ast.Attribute(value=target, attr='append', ctx=ast.Load()), args=[elt],
                                   keywords=[]))]
            elif type(node) == ast.SetComp:
                elt = node.elt
                temp = ast.Assign(targets=[target], value=ast.Set(elts=[]))
                loop_body = [ast.Expr(
                    value=ast.Call(func=ast.Attribute(value=target, attr='add', ctx=ast.Load()), args=[elt], keywords=[]))]
            elif type(node) == ast.DictComp:
                temp = ast.Assign(targets=[target], value=ast.Dict(keys=[], values=[]))
                loop_body = [ast.Assign(targets=[ast.Subscript(value=target, slice=ast.Index(value=node.key), ctx=ast.Store())],
                                        value=node.value)]
            for generator in reversed(generators):
                if generator.ifs:
                    cond = generator.ifs[0]
                    loop_body = [ast.For(target=generator.target, iter=generator.iter,
                                         body=[ast.If(test=cond, body=loop_body, orelse=[])], orelse=[], type_comment=None)]
                else:
                    loop_body = [
                        ast.For(target=generator.target, iter=generator.iter, body=loop_body, orelse=[], type_comment=None)]
            if node.lineno == node.end_lineno:
                ass_str = astor.to_source(temp)
                for_str = astor.to_source(loop_body[0])
                action_list.append(("Comp2For",'all',[node.lineno],ass_str+for_str,(temp,loop_body)) )
    return action_list

# tree = astor.code_to_ast.parse_file('exmple.py')
# visit(tree, [], 0)
# dic = findComp(tree)
# mutateComp2For(dic)

def findExprStmt(tree):
    ExpStmt_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.Expr and type(node.parent) != ast.Assign:
            ExpStmt_nodes.append(node)
    return ExpStmt_nodes

def mutateExprStmt2Assign(nodes,name_set):
    action_list=[]
    for node in nodes:
        name = generate_random_name(name_set)
        if node.lineno == node.end_lineno:
            if hasattr(node.parent, 'body'):
                if node in node.parent.body:
                    action_list.append(("ExprStmt2Assign",'all',[node.lineno],name+'='))
    # if len(action_list) >100:
    #     action_list = random.sample(action_list,100)
    return action_list

def findAugAssginStmt(tree):
    aug_Assgin_Stmt_nodes = []
    for node in ast.walk(tree):
        if type(node) == ast.AugAssign:
            aug_Assgin_Stmt_nodes.append(node)
    return aug_Assgin_Stmt_nodes

def mutateAugAssgin2Assgin(assgin_Stmts):
    action_list=[]
    for node in assgin_Stmts:
        new_assgin = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, right=node.value, op=node.op)
        )
        new_str = astor.to_source(new_assgin)
        if node.lineno == node.end_lineno:
            action_list.append(("AugAssgin2Assgin",'all',[node.lineno],new_str))
    return action_list

def findAssert(tree):
    assert_stmts = []
    for node in ast.walk(tree):
        if type(node) == ast.Assert:
            assert_stmts.append(node)
    return assert_stmts

def mutateAssert(nodes):
    action_list = []
    for node in nodes:
        new_test_list = notConditon(node.test)
        for new_test in new_test_list:
            new_node = ast.If(
                test=new_test,
                body=[ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError',ctx=ast.Load()),keywords=[],args=[node.msg]),cause=None)],
                orelse=[]
            )
            new_str = astor.to_source(new_node)
            if node.lineno == node.end_lineno:
                action_list.append(("Assert2If",'all',[node.lineno],new_str))
    return action_list

def findIfStmt(tree):
    if_stmts = []
    for node in ast.walk(tree):
        if type(node) == ast.If:
            if_stmts.append(node)
    return if_stmts

def mutateIfStmt(nodes):
    action_list = []
    for node in nodes:
        if len(node.orelse) == 0:
            continue
        if type(node.parent) == ast.If:
            if  len(node.parent.orelse) == 1 and type(node.parent.orelse[0]) == node:
                continue
        elif len(node.orelse) == 1 and type(node.orelse[0]) == ast.If:
            root_node = copy.copy(node)
            new_node = root_node
            new_node.test = random.choice(notConditon(new_node.test))
            new_node.body, new_node.orelse = new_node.orelse, new_node.body
            while len(new_node.body) == 1 and type(new_node.body[0]) == ast.If:
                new_node = new_node.body[0]
                new_node.test = random.choice(notConditon(new_node.test))
                new_node.body, new_node.orelse = new_node.orelse, new_node.body
                if type(new_node) != ast.If:
                    break
            new_node = root_node
            new_str = astor.to_source(new_node)
            action_list.append(("IfStmt2IfStmt",'all',[node.lineno,node.end_lineno],new_str))
        else:
            new_test_list = notConditon(node.test)
            for new_test in new_test_list:
                new_node = ast.If(
                    test=new_test,
                    body=node.orelse,
                    orelse=node.body
                )
                new_str = astor.to_source(new_node)
                action_list.append(("IfStmt2IfStmt", 'all', [node.lineno, node.end_lineno], new_str))
    return action_list

def findWhileStmt(tree):
    while_stmts = []
    for node in ast.walk(tree):
        if type(node) == ast.While:
            while_stmts.append(node)
    return while_stmts

def mutateWhileStmt(nodes):
    action_list = []
    for node in nodes:
        new_test_list = notConditon(node.test)
        for new_test in new_test_list:
            stop_line = ast.If(
                test = new_test,
                body=[ast.Break()],
                orelse=[]
            )
            new_node = ast.While(
                test=ast.Constant(value=True, kind=None),
                body=[stop_line]+node.body,
                orelse=node.orelse
            )
            new_str = astor.to_source(new_node)
            action_list.append(("WhileStmt", 'all', [node.lineno, node.end_lineno], new_str))
    return action_list

def findForStmt(tree):
    for_stmts = []
    for node in ast.walk(tree):
        if type(node) == ast.For:
            for_stmts.append(node)
    return for_stmts

def mutateForStmt(nodes,name_set):
    action_list = []
    for node in nodes:
        # use iter replace for loop
        name = generate_random_name(name_set)
        ass_line = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node.iter)
        try_scope = ast.Try(
            body=[ast.Assign(targets=[node.target], value=ast.Call(func=ast.Name(id='next',ctx=ast.Load()),args=ass_line.targets,keywords=[]))]+node.body,
            handlers=[ast.ExceptHandler(type=ast.Name(id='StopIteration', ctx=ast.Load()), name=None, body=[ast.Break()])],
            orelse=[],
            finalbody=[]
        )
        new_node = ast.While(
            test=ast.Constant(value=True, kind=None),
            body=[try_scope],
            orelse=node.orelse
        )
        new_ass_str = astor.to_source(ass_line)
        new_str = astor.to_source(new_node)
        action_list.append(("ForStmt", 'all', [node.lineno, node.end_lineno], new_ass_str+new_str))
    return action_list

def findWithStmt(tree):
    with_stmts = []
    for node in ast.walk(tree):
        if type(node) == ast.With:
            with_stmts.append(node)
    return with_stmts

def mutateWith2Try(nodes):
    action_list=[]
    for node in nodes:
        ctx_expr = node.items[0].context_expr
        optional_vars = node.items[0].optional_vars
        if optional_vars==None:
            continue
        body = node.body

        # 构造try语句
        try_node = ast.Try(
            body=body,
            handlers=[],
            orelse=[],
            finalbody=[]
        )

        # 构造with语句的退出代码
        exit_call = ast.Call(
            func=ast.Attribute(
                value=optional_vars,
                attr="__exit__",
                ctx=ast.Load()
            ),
            args=[
                ast.Name(
                    id="None",
                    ctx=ast.Load()
                ),
                ast.Name(
                    id="None",
                    ctx=ast.Load()
                ),
                ast.Name(
                    id="None",
                    ctx=ast.Load()
                )
            ],
            keywords=[]
        )
        exit_expr = ast.Expr(value=exit_call)

        # 构造try语句的finally块
        try_node.finalbody.append(exit_expr)

        # 构造with语句的进入代码
        enter_call = ast.Call(
            func=ast.Attribute(
                value=ctx_expr,
                attr="__enter__",
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        enter_expr = ast.Expr(value=ast.Assign(
            targets=[optional_vars],
            value=enter_call
        ))

        # 将try语句和进入代码合并为一个新的块
        ctx_str = astor.to_source(enter_expr) + astor.to_source(try_node)
        action_list.append(("With2Try", 'all', [node.lineno, node.end_lineno], ctx_str))
    return action_list

def findScope2Func(tree):
    zip_Scope = []
    for node in ast.walk(tree):
        if hasattr(node, 'body'):
            if len(node.body) > 5:
                start = random.randint(0, len(node.body))
                if start + 5 < len(node.body):
                    end = random.randint(start+2, start + 5)
                else:
                    end = random.randint(start, len(node.body))
                zip_Scope.append(node.body[start:end])
    return zip_Scope

def mutateZipScope(scopes,name_set):
    action_list = []
    for scope in scopes:
        func_name = generate_random_name(name_set)
        load_var = set()
        load_name = set()
        arg_var=set()
        arg_name=set()
        store_var = set()
        store_name = set()
        start_line = scope[0].lineno
        end_line = scope[-1].end_lineno
        return_var=[]
        for line in scope:
            return_var+=line.targets
            for node in ast.walk(line):
                if type(node) == ast.Name and isinstance(node.ctx, ast.Load):

                    if type(node.parent) == ast.Call:
                        if node.parent.func==node:
                            continue
                    if type(node.parent) == ast.ExceptHandler:
                        if node.parent.type==node:
                            continue
                    if node.id not in load_name:
                        load_var.add(node)
                        load_name.add(node.id)
                    if node.id not in arg_name:
                        arg_var.add(ast.arg(arg=node.id, annotation=None))
                        arg_name.add(node.id)
                if type(node) == ast.Name and isinstance(node.ctx, ast.Store):
                    if node.id not in store_name:
                        store_name.add(node.id)
                        store_var.add(node)
        return_node = ast.Return(value=ast.Tuple(elts=return_var,ctx = ast.Load()))
        new_node = ast.Assign(targets = [ast.Tuple(elts=return_var,ctx = ast.Store())], value = ast.Call(func=ast.Name(id=func_name,ctx=ast.Load()),args=load_var,keywords=[]))
        use_str = astor.to_source(new_node)
        func_node = ast.FunctionDef(name=func_name, args=ast.arguments(args=arg_var, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=scope+[return_node], decorator_list=[], returns=None)
        def_str = astor.to_source(func_node)
        action_list.append(("Scope2Func", 'all', [start_line, end_line], def_str+use_str))
    return action_list

def zipAssgin(tree):
    zip_Assgin = []
    for node in ast.walk(tree):
        if hasattr(node, 'body'):
            i = 0
            while i < len(node.body):
                sub_list = []
                if type(node.body[i]) == ast.Assign and len(node.body[i].targets) == 1:
                    sub_list.append(node.body[i])
                    while i+1 < len(node.body) and type(node.body[i+1]) == ast.Assign and len(node.body[i].targets) == 1:
                        sub_list.append(node.body[i+1])
                        i += 1
                i += 1
            if len(sub_list) > 1:
                zip_Assgin.append(sub_list)
    return zip_Assgin

def mutateZipAssgin(nodes_set):
    action_list = []
    for nodes in nodes_set:
        targets = []
        values = []
        for node in nodes:
            targets.append(node.targets[0])
            values.append(node.value)
        new_node = ast.Assign(targets=[ast.Tuple(elts=targets,ctx=ast.Store())], value=ast.Tuple(elts=values, ctx=ast.Load()))
        new_str = astor.to_source(new_node)
        action_list.append(("ZipAssgin", 'all', [nodes[0].lineno, nodes[-1].end_lineno], new_str))
    return action_list

def addGarbageCode(tree):
    return

def analyzeCode(tree):
    if_expr_nodes = []
    assign_nodes = []
    lambda_nodes = []
    comp_nodes = []
    ExpStmt_nodes = []
    aug_Assgin_Stmt_nodes = []
    assert_stmts = []
    if_stmts = []
    while_stmts = []
    for_stmts = []
    with_stmts = []
    zip_Scope = []
    zip_Assgin = []
    add_GarbageCode = []
    for node in ast.walk(tree):
        if type(node) == ast.IfExp:
            if_expr_nodes.append(node)
        if type(node) == ast.Assign:
            if type(node.value) == ast.List or type(node.value) == ast.Tuple or type(node.value) == ast.Set or type(
                    node.value) == ast.Dict or type(node.value) == ast.Call:
                assign_nodes.append(node)
        if type(node) == ast.Lambda:
            lambda_nodes.append(node)
        if type(node) == ast.ListComp or type(node) == ast.SetComp or type(node) == ast.DictComp:
            comp_nodes.append(node)
        if type(node) == ast.Expr and type(node.parent) != ast.Assign and type(node.parent) != ast.AugAssign and type(
                node.parent) != ast.Module:
            ExpStmt_nodes.append(node)
        if type(node) == ast.AugAssign:
            aug_Assgin_Stmt_nodes.append(node)
        if type(node) == ast.Assert:
            assert_stmts.append(node)
        if type(node) == ast.If:
            if_stmts.append(node)
        if type(node) == ast.While:
            while_stmts.append(node)
        if type(node) == ast.For:
            for_stmts.append(node)
        if type(node) == ast.With:
            with_stmts.append(node)
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
                        zip_Scope.append(node.body[start:end])
        if hasattr(node, 'body') and isinstance(node.body, list):
            for sub_node in node.body:
                add_GarbageCode.append(sub_node.lineno)
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
            if len(sub_list) > 1:
                zip_Assgin.append(sub_list)
    return if_expr_nodes,assign_nodes,lambda_nodes,comp_nodes,ExpStmt_nodes,aug_Assgin_Stmt_nodes,assert_stmts,if_stmts,while_stmts,for_stmts,with_stmts,zip_Scope,zip_Assgin,add_GarbageCode


def process_node(file_ct,file_name,src_code, action_list,scope_list,add_code,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num = 0
    code = '\n'.join(src_code)
    token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
    out_tokens = []
    prev_eol = False
    start_line = 0
    try:
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
                start_line = end[0] // 2 + 1
                break
    except:
        print(file_name)
        return
    if len(out_tokens) < 900:
        return
    # start_line = len(src_code)-1
    for line in range(start_line, len(src_code)):
        sub_action_list = []
        tg_line = line
        code = ''.join(src_code)
        token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
        orig_tokens = []
        prev_eol = False
        str_ct=0
        tgt_tokens=[]
        lineno_list=[]
        try:
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
                if start[0] > tg_line+ 1 and len(tgt_tokens) == 0:
                    tgt_tokens=[]
                    tg_line += 1
                if start[0] > tg_line + 1 and len(tgt_tokens)==1 and tgt_tokens[0]=="<EOL>":
                    tgt_tokens = []
                    tg_line += 1
                if toknum == STRING:
                    add_token = process_string(tokval)
                    if start[0] <= tg_line:
                        orig_tokens.append(add_token)
                        lineno_list.append(start[0])
                    elif start[0] > tg_line and start[0] <= tg_line + 1:
                        tgt_tokens.append(add_token)

                    prev_eol = False
                elif toknum == NUMBER:
                    if tokval in lits['num']:
                        if start[0] <= tg_line:
                            orig_tokens.append(f"<NUM_LIT:{tokval}>")
                            lineno_list.append(start[0])
                        elif start[0] > tg_line and start[0] <= tg_line + 1:
                            tgt_tokens.append(f"<NUM_LIT:{tokval}>")

                    else:
                        if start[0] <= tg_line:
                            orig_tokens.append(f"<NUM_LIT>")
                            lineno_list.append(start[0])
                        elif start[0] > tg_line and start[0] <= tg_line + 1:
                            tgt_tokens.append(f"<NUM_LIT>")

                    prev_eol = False
                elif toknum in [NEWLINE, NL]:
                    if not prev_eol:
                        if start[0] <= tg_line:
                            orig_tokens.append("<EOL>")
                            lineno_list.append(start[0])
                        elif start[0] > tg_line and start[0] <= tg_line + 1:
                            tgt_tokens.append("<EOL>")

                        prev_eol = True
                elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                    continue
                else:
                    if start[0] <= tg_line:
                        orig_tokens.append(tokval)
                        lineno_list.append(start[0])
                    elif start[0] > tg_line and start[0] <= tg_line+ 1:
                        tgt_tokens.append(tokval)
                    prev_eol = False

                if len(orig_tokens) > 900:
                    orig_tokens.pop(0)
                    lineno_list.pop(0)
                if start[0] > tg_line+ 1 and len(tgt_tokens) > 0:
                    if len(tgt_tokens)==1 and tgt_tokens[0]=="<EOL>":
                        continue
                    break
        except:
           print(file_name)
           continue
        if len(lineno_list)!=0:
            farthest_line = lineno_list[0]
        else:
            farthest_line = 0
        if orig_tokens[0] == "<EOL>":
            orig_tokens = orig_tokens[1:]
        try:
            if tgt_tokens[0] == "<EOL>":
                tgt_tokens = tgt_tokens[1:]
            if tgt_tokens[-1] == "<EOL>":
                tgt_tokens = tgt_tokens[:-1]
        except:
            continue
        orig_tokens = ["<s>"] + orig_tokens
        if len(orig_tokens) > 900:
            orig_tokens = orig_tokens[-900:]
        orig_str = " ".join(orig_tokens)
        tgt_str = " ".join(tgt_tokens)
        temp_dict = {
            'id': str_ct,
            'input': orig_str,
            'gt': tgt_str,
            'action': None,
        }
        out_line = json.dumps(temp_dict)

        if not os.path.exists(save_path + '/' +str(file_ct)):
            os.makedirs(save_path + '/' +str(file_ct))
        wf = open(save_path + '/' +str(file_ct)+'/' + str(file_ct) + '_' + str(tg_line)+'.jsonl', 'w')
        wf.write(out_line + '\n')
        for action_ in action_list:
            # if action_[2][-1] <= tg_line:
            if action_[2][-1] <= tg_line and action_[2][0] >= farthest_line:
                sub_action_list.append(action_)
        for action in sub_action_list:
            line_add = 0
            if action[0]=='IfExpr2Stmt':
                act_line = action[2][0]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line-1] + [new_line] + src_code[act_line:]
                line_add = len(re.findall('\n',action[3]))-len(re.findall('\n',src_code[act_line-1]))
            elif action[0]=='negate_if_expr':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(
                    re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='AssginAddLine':
                act_line = action[2][0]
                new_line = src_code[act_line-1]
                enter_start = re.findall('\n',new_line)
                comment = ''
                if new_line.find('#')!=-1:
                    new_line_ = new_line[:new_line.find('#')]
                    comment = new_line[new_line.find('#'):]
                    new_line= new_line_
                if '(\n' in new_line and '\n)' in new_line:
                    new_line =new_line.replace('(\n', '(')
                    new_line =new_line.replace('\n)', ')')
                elif '(' in new_line and ')' in new_line:
                    new_line =new_line.replace('(', '(\n')
                    new_line =new_line.replace(')', '\n)')
                if '{\n' in new_line and '\n}' in new_line:
                    new_line =new_line.replace('{\n', '{')
                    new_line =new_line.replace('\n}', '}')
                elif '{' in new_line and '}' in new_line:
                    new_line =new_line.replace('{', '{\n')
                    new_line =new_line.replace('}', '\n}')
                if '[\n' in new_line and '\n]' in new_line:
                    new_line =new_line.replace('[\n', '[')
                    new_line =new_line.replace('\n]', ']')
                elif '[' in new_line and ']' in new_line:
                    new_line =new_line.replace('[', '[\n')
                    new_line =new_line.replace(']', '\n]')
                enter_stop = re.findall('\n',new_line)
                line_add = len(enter_stop) - len(enter_start)
                new_src_code = src_code[:act_line-1] + [new_line+comment] + src_code[act_line:]
            elif action[0]=='Lambda2Func':
                act_line = action[2][0]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                if action[4][0] == action[4][1]:
                    repstr = src_code[action[4][0]-1][:action[4][-2]] + action[5] + src_code[action[4][0]-1][action[4][-1]:]
                    ctx_ = src_code[act_line-1:]
                    ctx_[action[4][0]-act_line] = repstr
                    new_src_code = src_code[:act_line-1]  + [new_line] + ctx_
                else:
                    repstr=src_code[action[4][0] - 1][:action[4][-2]] + action[5] + src_code[action[4][1] - 1][action[4][3]-1:]
                    ctx_=src_code[act_line-1:]
                    ctx = src_code[:action[4][0]-act_line]+[repstr]+src_code[action[4][1]-act_line+1:]
                    src_code[:act_line - 1] + [new_line] + [
                        ] + src_code[action[4][1]:]
                    new_src_code = src_code[:act_line - 1] + [new_line] + ctx_
                line_add = len(re.findall('\n', action[3])) + len(re.findall('\n',repstr))- len(re.findall('\n', ''.join(src_code[action[4][0]-1:action[4][1]])))
            elif action[0]=='Comp2For':
                act_line = action[2][0]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line-1] + [new_line] + src_code[act_line:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', src_code[act_line - 1]))
            elif action[0]=='ExprStmt2Assign':
                act_line = action[2][0]
                st = re.findall('\S',src_code[act_line-1])
                st = src_code[act_line-1].find(st[0])
                new_src_code = src_code[:act_line-1] + [src_code[act_line-1][:st]+action[3]+src_code[act_line-1][st:]] + src_code[act_line:]
                line_add = 0
            elif action[0]=='AugAssgin2Assign':
                act_line = action[2][0]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line-1] + [new_line] + src_code[act_line:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', src_code[act_line - 1]))
            elif action[0]=='Assert2If':
                act_line = action[2][0]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line-1] + [new_line] + src_code[act_line:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', src_code[act_line - 1]))
            elif action[0]=='IfStmt2IfStmt':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line-1][:st]+action[3].replace('\n','\n'+src_code[act_line-1][:st])[:-len(src_code[act_line-1][:st])]
                new_src_code = src_code[:act_line-1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='WhileStmt':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='ForStmt':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='With2Try':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='Scope2Func':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='ZipAssgin':
                act_line = action[2][0]
                act_end = action[2][1]
                st = re.findall('\S', src_code[act_line - 1])
                st = src_code[act_line - 1].find(st[0])
                new_line = src_code[act_line - 1][:st] + action[3].replace('\n', '\n' + src_code[act_line - 1][:st])[
                                                         :-len(src_code[act_line - 1][:st])]
                new_src_code = src_code[:act_line - 1] + [new_line] + src_code[act_end:]
                line_add = len(re.findall('\n', action[3])) - len(re.findall('\n', ''.join(src_code[act_line - 1:act_end])))
            elif action[0]=='GarbageCode':
                if action[1]=='all':
                    act_line = random.choice(add_code)
                    while act_line-1 >= tg_line:
                        act_line = random.choice(add_code)
                    near = 10000
                    scope={}
                    for key,value in scope_list.items():
                        if act_line>value['##scope##'][0] and act_line<value['##scope##'][1]:
                            if value['##scope##'][1]-value['##scope##'][0]<near:
                                scope = value
                    ass_list=set()
                    use_list=set()
                    for key in scope.keys():
                        if key == "##scope##":
                            continue
                        for idx,_ in scope[key]['assign']:
                            if idx<act_line:
                                ass_list.add(key)
                    for key in scope.keys():
                        min_use = 10000
                        min_ass = 10000
                        if key == "##scope##":
                            continue
                        for idx,_ in scope[key]['use']:
                            if idx>act_line:
                                if idx < min_use:
                                    min_use = idx
                        for idx,_ in scope[key]['assign']:
                            if idx > act_line:
                                if idx < min_ass:
                                    min_ass = idx
                        if min_ass <min_use:
                            use_list.add(key)
                    ass_list = list(ass_list)
                    use_list = list(use_list)
                    if len(ass_list)>0 and len(use_list)>0:
                        new_line = astor.to_source(ast.If(test=ast.Compare(ops=[ast.Eq()],left=ast.Name(id=ass_list[0],ctx=ast.Load()),comparators=[ast.Name(id=ass_list[0],ctx=ast.Load())]), body=[ast.Assign(targets=[ast.Name(id=use_list[0],ctx=ast.Load())],value=ast.Name(id=ass_list[-1],ctx=ast.Load()))], orelse=[]))
                    elif len(use_list)>0:
                        new_line = astor.to_source(ast.If(test=ast.Constant(value='True'), body=[ast.Assign(targets=[ast.Name(id=use_list[0],ctx=ast.Load())],value=ast.Constant(value=random.randint(0,99)))], orelse=[]))
                    elif len(ass_list)>0:
                        new_line = astor.to_source(ast.If(test=ast.Compare(ops=[ast.Eq()],
                                                                           left=ast.Name(id=ass_list[0], ctx=ast.Load()),
                                                                                         comparators=[
                                                                                             ast.Name(id=ass_list[0],
                                                                                                      ctx=ast.Load())]),
                                                          body=[ast.Pass],
                                                          orelse=[]))
                    else:
                        new_line = astor.to_source(ast.If(test=ast.Constant(value='True'), body=[ast.Pass], orelse=[]))
                    up_line = act_line-1
                    while len(re.findall('\S', src_code[up_line]))==0:
                        up_line=up_line-1
                    st = re.findall('\S', src_code[up_line])
                    new_enter = new_line.find('\n')
                    down_line = act_line
                    while len(re.findall('\S', src_code[down_line]))==0:
                        down_line=down_line+1
                    if st[-1]!=':':
                        st = src_code[up_line].find(st[0])
                        new_src_code = src_code[:act_line] + [src_code[up_line][:st]+new_line[:new_enter+1]+src_code[up_line][:st]+new_line[new_enter+1:]] + src_code[act_line:]
                    else:
                        st = re.findall('\S', src_code[down_line])
                        st = src_code[down_line].find(st[0])
                        new_src_code = src_code[:act_line] + [src_code[down_line][:st] + new_line[:new_enter+1]+src_code[down_line][:st]+new_line[new_enter+1:]] + src_code[
                                                                                                        act_line:]
            else:
                continue
            # continue
            code = ''.join(new_src_code)
            token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
            out_tokens = []
            prev_eol = False
            break_flag=0
            try:
                for toknum, tokval, start, end, line in token_gen:
                    if start[0] > tg_line + line_add:
                        break_flag=1
                        break
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
            except:
                if break_flag!=1:
                    print('error',file=open('error.txt','a'))
                    print(file_name,file=open('error.txt','a'))
                    # print(code,file=open('error.txt','a'))
                    # print(tg_line,file=open('error.txt','a'))
                    # print(line_add,file=open('error.txt','a'))
                    print(action,file=open('error.txt','a'))
                    continue
            if out_tokens[0] == "<EOL>":
                out_tokens = out_tokens[1:]
            src_tokens = ["<s>"] + out_tokens
            if len(src_tokens) > 900:
                src_tokens = src_tokens[-900:]
            src_str = ' '.join(src_tokens)
            str_ct +=1
            temp_dict = {
                'id': str_ct,
                'input': src_str,
                'gt': tgt_str,
                'action': action,
            }
            out_line = json.dumps(temp_dict)
            wf.write(out_line + '\n')

    return None


def process(job_i,dataset):
    for file_ct,file_name in tqdm(dataset,desc='job'+str(job_i)):
        code = open('./dataset/py150/'+file_name).readlines()
        tree = astor.code_to_ast.parse_file('./dataset/py150/'+file_name)
        visit(tree, [], 0)
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",cache_dir= './bert')
        scope_list, name_set = findIdentifier(tree)

        # mutate_var(scope_list, tokenizer)
        # use analyzer to find the nodes we want to mutate
        if_Expr, assgin_Expr, lambda_Expr, comp_Expr, expr_Stmts, assgin_Stmts, assert_Stmts, if_Stmts, while_Stmts, for_Stmts, with_Stmts, zip_Scope, zip_Assgin, add_gbg_code = analyzeCode(
            tree)
        action_list = []
        action_list += mutateIfExpr2Stmt(if_Expr)
        action_list += negate_if_expr(if_Expr)
        action_list += mutateAssginAddLine(assgin_Expr)
        action_list += mutateLambda2func(lambda_Expr, name_set)
        action_list += mutateComp2For(comp_Expr)
        action_list += mutateExprStmt2Assign(expr_Stmts, name_set)
        action_list += mutateAugAssgin2Assgin(assgin_Stmts)
        action_list += mutateAssert(assert_Stmts)
        action_list += mutateIfStmt(if_Stmts)
        action_list += mutateWhileStmt(while_Stmts)
        action_list += mutateForStmt(for_Stmts, name_set)
        action_list += mutateWith2Try(with_Stmts)
        action_list += mutateZipScope(zip_Scope, name_set)
        action_list += mutateZipAssgin(zip_Assgin)
        action_list += [('GarbageCode', 'all', [0])] * 3
        # action_list += [('GarbageCode', 'near', [0])]
        # print(action_list)
        addGarbageCode(tree)
        process_node(file_ct,file_name,code, action_list, scope_list, add_gbg_code,'dataset/py150/mutated_line/')
    return None
# test = "dataset/emedvedev-attention-ocr/0/src.py"
# process(1,[(0,test)])

if __name__ == '__main__':
    # Read the AST from stdin
    index = open("dataset/py150/index.jsonl", "r").readlines()
    dataset= []
    for line in index:
        line = json.loads(line)
        file_name = line['path']
        file_ct = line['ct']
        dataset.append((file_ct,file_name))

    # raw_set=raw_set[14580:]
    job_num = 100
    devide_list = [[] for i in range(job_num)]
    for i, item in enumerate(dataset):
        devide_list[i % job_num].append(item)
    Parallel(n_jobs=job_num)(
        delayed(process)(i, code_set) for (i, code_set) in enumerate(devide_list))
    # graph.render("test", view=True)






