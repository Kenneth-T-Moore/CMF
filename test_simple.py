from __future__ import division
from framework import *


class VarA(ImplicitSystem):

    def _declare(self):
        self._declare_variable('a', val=1.0)
        self._declare_argument('b')

    def apply_F(self):
        v = self.vec
        a = v['u']('a')[0]
        b = v['p']('b')[0]
        f = b - a**3
        #f = 2*a + 3*b - 5
        v['f']('a')[0] = f

    def apply_dFdpu(self, arguments):
        v = self.vec
        a = v['u']('a')[0]
        b = v['p']('b')[0]
        df = v['df']('a')
        da = v['du']('a')
        db = v['dp']('b')
        dFda = -3*a**2
        dFdb = 1
        #dFda = 2
        #dFdb = 3
        if self.mode == 'fwd':
            df[0] = 0.0
            if self.get_id('a') in arguments:
                df[0] += dFda * da[0]
            if self.get_id('b') in arguments:
                df[0] += dFdb * db[0]
        elif self.mode == 'rev':
            if self.get_id('a') in arguments:
                da[0] = dFda * df[0]
            if self.get_id('b') in arguments:
                db[0] = dFdb * df[0]
                

class VarB(ImplicitSystem):

    def _declare(self):
        self._declare_variable('b', val=1.0)
        self._declare_argument('a')

    def apply_F(self):
        v = self.vec
        a = v['p']('a')[0]
        b = v['u']('b')[0]
        f = b - numpy.exp(-a)
        #f = a + 2*b - 2
        v['f']('b')[0] = f

    def apply_dFdpu(self, arguments):
        v = self.vec
        a = v['p']('a')[0]
        b = v['u']('b')[0]
        df = v['df']('b')
        da = v['dp']('a')
        db = v['du']('b')
        dFda = numpy.exp(-a)
        dFdb = 1
        #dFda = 1
        #dFdb = 2
        if self.mode == 'fwd':
            df[0] = 0.0
            if self.get_id('a') in arguments:
                df[0] += dFda * da[0]
            if self.get_id('b') in arguments:
                df[0] += dFdb * db[0]
        elif self.mode == 'rev':
            if self.get_id('a') in arguments:
                da[0] = dFda * df[0]
            if self.get_id('b') in arguments:
                db[0] = dFdb * df[0]


main = SerialSystem('main', subsystems=[
        VarA('a'),
        VarB('b'),
        ]).setup()


print main.compute()
print main.compute_derivatives('fwd', 'a', output=False)
print main.compute_derivatives('fwd', 'b', output=False)
print main.compute_derivatives('rev', 'a', output=False)
print main.compute_derivatives('rev', 'b', output=False)

main.check_derivatives_all(fwd=True)
