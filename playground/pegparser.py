"""a recursive descent parser using pypeg"""
from pypeg2 import *

r = re.compile

class IOFunc(str):
    grammar = r(r'read|write')

class Path(str):
    grammar = r(r'"([^"]+)"')

class L1Opr(str):
    grammar = r(r'[*/]')

class L2Opr(str):
    grammar = r(r'[+-]')

class Number(str):
    grammar = r(r'\d+')

class Var(str):
    grammar = r(r'[a-zA-Z]\w*')

class Word(List):
    pass

class Term(List):
    pass

class Expr(List):
    pass

class Parameters(List):
    grammar = csl(Var)

class Func(List):
    grammar = word, '(', Parameters, ')'

Term.grammar = Word, maybe_some(L1Opr, Term)

Expr.grammar = Term, maybe_some(L2Opr, Expr)

Word.grammar = [
        Func,
        Var,
        ('(', Term, ')'),
        ('(', Expr, ')'),
    ], optional("**", Number)

class Instruction(List):
    grammar = [
        (r(r'del'), Var),
        (IOFunc, '(', Var, ',', Path, ')', endl),
        (Var, '=', Expr, endl),
    ]

class Program(List):
    grammar = maybe_some(Instruction)

with open('prog.txt', 'r') as f:
    src = f.read()
f = parse(src, Program)

def iprint(x, ident=''):
    if isinstance(x, str):
        print(ident, x, sep='')
    else:
        if len(x) == 1:
            iprint(x[0], ident)
        else:
            for v in x:
                iprint(v, ident+'  ')

iprint(f)
from IPython import embed; embed()

