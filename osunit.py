#!/bin/python

import numpy as np
from fractions import Fraction as frac
import re


class osunits:
    name = [ 'm_e', 'c', '\omega_p', 'e', 'n_0' ]
    xtrnum = re.compile(r"(?<=\^)\d+|(?<=\^{).*?(?=})")


    def __init__( self, s ):
        sl = s.split()
        positive = True
        self.power = np.array([frac(0), frac(0), frac(0), frac(0), frac(0)])
        while sl:
            ss = sl.pop(0)
            if ss == '/':
                positive = False
                continue
            for p, n in enumerate(osunits.name):
                if n in ss:
                    res = osunits.xtrnum.findall(ss)  # extract numbers
                    if res:
                        self.power[p] = frac(res[0]) if positive else -frac(res[0])
                    else:
                        self.power[p] = frac(1, 1) if positive else frac(-1, 1)
                    break
            else:
                raise KeyError('Unknown unit: ' + re.findall(r'\w+', ss)[0])

    def tex( self ):
        return '$' + self.__str__() + '$'

    def __mul__( self, other ):
        res = osunits('')
        res.power = self.power + other.power
        return res

    def __truediv__( self, other ):
        res = osunits('')
        res.power = self.power - other.power
        return res

    def __pow__( self, other, modulo=1 ):
        res = osunits('')
        res.power = self.power * frac(other)
        return res

    def __eq__( self, other ):
        return ( self.power == other.power ).all()

    def __str__( self ):
        disp = ''
        for n, p in zip(osunits.name, self.power):
            if p == 0:
                continue
            elif p == 1:
                disp = disp + n + ' '
            else:
                disp = disp + n + '^{' + str(p) + '}'
        if not disp:
            disp = 'a.u.'
        return disp
    

# tests
if __name__ == '__main__':
    print(osunits("e \omega_p^2 / c"))
    print(osunits('m_e') * osunits('c'))
    print(osunits('m_e') / osunits('m_e'))
    print(osunits('m_e')**-1.5)
    print(osunits('m_e')**"5/7")
    print(osunits('m_e')**(5/7))  # We should not use this notation when the power has too many decimal digits 
    print(osunits('n_0') == osunits('n_0'))

    a = osunits('n_0')
    b = a
    print(b)

    osunits('n_0').tex()
    print(osunits("m_e e / \omega_p c eua^2"))  # this will not raise an error
    print(osunits("m_e e / \omega_p c ua^2"))  # but this will, these are low impact corner cases which probaly won't be fixed
