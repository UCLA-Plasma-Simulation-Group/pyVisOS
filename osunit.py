#!/usr/bin/env python

"""osunit.py: Define units used in OSIRIS output."""

import numpy as np
from fractions import Fraction as frac
import re


class OSUnits:
    name = ['m_e', 'c', '\omega_p', 'e', 'n_0']
    xtrnum = re.compile(r"(?<=\^)\d+|(?<=\^{).*?(?=})")

    def __init__(self, s):
        """
        :param s: string notation of the units. there should be whitespace around quantities and '/' dividing quantities
        """
        self.power = np.array([frac(0), frac(0), frac(0), frac(0), frac(0)])
        # if isinstance(s, bytes):
        #     s = s.decode("utf-8")
        if 'a.u.' != s:
            sl = s.split()
            nominator = True
            while sl:
                ss = sl.pop(0)
                if ss == '/':
                    nominator = False
                    continue
                for p, n in enumerate(OSUnits.name):
                    if n == ss[0:len(n)]:
                        res = OSUnits.xtrnum.findall(ss)  # extract numbers
                        if res:
                            self.power[p] = frac(res[0]) if nominator else -frac(res[0])
                        else:
                            self.power[p] = frac(1, 1) if nominator else frac(-1, 1)
                        break
                else:
                    raise KeyError('Unknown unit: ' + re.findall(r'\w+', ss)[0])

    def tex(self):
        return '$' + self.__str__() + '$'

    def limit_denominator(self, max_denominator=64):
        """call fractions.Fraction.limit_denominator method for each base unit"""
        self.power = np.array([u.limit_denominator(max_denominator=max_denominator) for u in self.power])

    def __mul__(self, other):
        res = OSUnits('')
        res.power = self.power + other.power
        return res

    def __truediv__(self, other):
        res = OSUnits('')
        res.power = self.power - other.power
        return res

    def __pow__(self, other, modulo=1):
        res = OSUnits('')
        res.power = self.power * frac(other)
        return res

    def __eq__(self, other):
        return (self.power == other.power).all()

    def __str__(self):
        disp = ''.join(['' if p == 0 else n+" " if p == 1 else ''.join([n, '^{', str(p), '} '])
                        for n, p in zip(OSUnits.name, self.power)])
        if not disp:
            return 'a.u.'
        return disp
    

# tests
if __name__ == '__main__':
    print([OSUnits("e \omega_p^2 / c")])
    print(OSUnits('m_e') * OSUnits('c'))
    print(OSUnits('m_e') / OSUnits('m_e'))
    print(OSUnits('m_e')**-1.5)
    print(OSUnits('m_e')**"5/7")
    not_good = OSUnits('m_e')**(5/7)  # We should not use this notation when the power has too many decimal digits
    print(not_good)
    not_good.limit_denominator()  # we do provide a way to fix the **(5/7) problem
    print(not_good)
    print(OSUnits('n_0') == OSUnits('n_0'))

    a = OSUnits('n_0')
    b = a
    print(b.tex())

    print(OSUnits("m_e e / \omega_p c eua^2"))  # this will not raise an error but
    print(OSUnits("m_e e / \omega_p c ua^2"))  # this will fail, these corner cases probably won't be fixed
