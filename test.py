from __future__ import division
from system import *


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
        da = v['du']('a')[0]
        db = v['dp']('b')[0]
        dFda = -3*a**2
        dFdb = 1
        #dFda = 2
        #dFdb = 3
        v['df']('a')[0] = dFda * da + dFdb * db


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
        da = v['dp']('a')[0]
        db = v['du']('b')[0]
        dFda = numpy.exp(-a)
        dFdb = 1
        #dFda = 1
        #dFdb = 2
        v['df']('b')[0] = dFda * da + dFdb * db


main = SerialSystem('main', subsystems=[
        VarA('a'),
        VarB('b'),
        ]).setup()


print main.compute()
print main.compute_derivatives('fwd', 'a')

if main('a').comm is not None:
    print 'a:', main('a').check_derivatives()
if main('b').comm is not None:
    print 'b:', main('b').check_derivatives()
