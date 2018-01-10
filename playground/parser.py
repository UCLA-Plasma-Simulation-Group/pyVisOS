#!/usr/bin/python3

class Parser:
    def __init__(self):
        pass

    def parse(self, expr):
        assert isinstance(expr, str), 'expression must be str'
        print('parsing:', expr)
        expr = expr + '\0'
        opr = []
        var = []
        name = ''
        last_alpha = False
        for c in expr:
            if c.isalpha():
                name += c
            elif c == '(':
                if last_alpha:
                    opr.append('func_'+name)
                    name = ''
                else:
                    opr.append(c)
            elif c == ')':
                if name:
                    var.append(name)
                    name = ''
                while opr[-1] != '(' and not opr[-1].startswith('func_'):
                    y = var.pop()
                    o = opr.pop()
                    x = var.pop()
                    print(x, o, y)
                    var.append(x+o+y)
                if opr[-1] == '(':
                    x = var.pop()
                    var.append('('+x+')')
                else:
                    x = var.pop()
                    print(opr[-1], x)
                    var.append(opr[-1]+'('+x+')')
                opr.pop()
            else:
                if name:
                    var.append(name)
                    name = ''
                # TODO: priority
                while len(opr) > 0 and opr[-1] != '(' and not opr[-1].startswith('func_'):
                    y = var.pop()
                    o = opr.pop()
                    x = var.pop()
                    print(x, o, y)
                    var.append(x+o+y)
                opr.append(c)
            last_alpha = c.isalpha()


if __name__ == '__main__':
    parser = Parser()
    parser.parse('x*fft(a+b+c-d)')
    # parser.parse('*fft(a+b+c-d)')
    parser.parse('*)fft(a+b+c-d)')

