from __future__ import division
from framework import *


class Var_yi(ImplicitSystem):

    def _declare(self):
        self._declare_variable(['yi',-1])
        self._declare_argument('x0')
        self._declare_argument(['xi',-1])
        self._declare_argument(['zi',-1])

    def apply_F(self):
        self._nln_init()
        p, u, f = self.vec['p'], self.vec['u'], self.vec['f']
        x0 = p('x0')[0]
        xi = p(['xi',-1])[0]
        ai = p(['zi',-1])[0]
        yi = u(['yi',-1])[0]
        f(['yi',-1])[0] = yi - xi*ai**3 + x0*xi**2 - 1
        self._nln_final()

    def apply_dFdpu(self, arguments):
        self._lin_init()
        p, u, f = self.vec['p'], self.vec['u'], self.vec['f']
        dp, du, df = self.vec['dp'], self.vec['du'], self.vec['df']
        x0 = p('x0')[0]
        xi = p(['xi',-1])[0]
        ai = p(['zi',-1])[0]
        yi = u(['yi',-1])[0]
        dx0 = dp('x0')[0]
        dxi = dp(['xi',-1])[0]
        dai = dp(['zi',-1])[0]
        dyi = du(['yi',-1])[0]
        dfdyi = 1.0
        dfdxi = -ai**3
        dfdai = -3*ai**2*xi
        dfdx0 = xi**2
        dfdxi = 2*xi*x0
        df(['yi',-1])[0] = dfdyi*dyi + dfdxi*dxi + dfdai*dai + dfdx0*dx0 + dfdxi*dxi
        self._lin_final()
        

class Var_zi(ImplicitSystem):
    
    def _declare(self):
        self._declare_variable(['zi',-1], val=1)
        self._declare_argument('x0')
        self._declare_argument(['xi',-1])
        self._declare_argument(['yi',-1])
        if self.comm.rank == 0:
            self._declare_argument(['zi',-1],[1])
        elif self.comm.rank == 1:
            self._declare_argument(['zi',-1],[0])

    def apply_F(self):
        self._nln_init()
        p, u, f = self.vec['p'], self.vec['u'], self.vec['f']
        x0 = p('x0')[0]
        xi = p(['xi',-1])[0]
        yi = p(['yi',-1])[0]
        if self.comm.rank == 0:
            ai = u(['zi',-1])[0]
            bi = p(['zi',-1])[0]
            f(['zi',-1])[0] = numpy.exp(-ai) - yi/xi - x0
        elif self.comm.rank == 1:
            ai = p(['zi',-1])[0]
            bi = u(['zi',-1])[0]
            f(['zi',-1])[0] = ai + bi
        self._nln_final()

    def apply_dFdpu(self, arguments):
        self._lin_init()
        p, u, f = self.vec['p'], self.vec['u'], self.vec['f']
        dp, du, df = self.vec['dp'], self.vec['du'], self.vec['df']
        x0 = p('x0')[0]
        xi = p(['xi',-1])[0]
        yi = p(['yi',-1])[0]
        dx0 = dp('x0')[0]
        dxi = dp(['xi',-1])[0]
        dyi = dp(['yi',-1])[0]
        if self.comm.rank == 0:
            ai = u(['zi',-1])[0]
            bi = p(['zi',-1])[0]
            dai = du(['zi',-1])[0]
            dbi = dp(['zi',-1])[0]
            dfdai = -numpy.exp(-ai)
            dfdyi = -1/xi
            dfdxi = yi/xi**2
            dfdx0 = -1
            df(['zi',-1])[0] = dfdai*dai + dfdyi*dyi + dfdxi*dxi + dfdx0*dx0
        elif self.comm.rank == 1:
            ai = p(['zi',-1])[0]
            bi = u(['zi',-1])[0]
            dai = dp(['zi',-1])[0]
            dbi = du(['zi',-1])[0]
            df(['zi',-1])[0] = dai + dbi
        self._lin_final()

main = \
    SerialSystem('main', subsystems=[
        IndVar('x0', val=10.0, size=1),
        ParallelSystem('pts', subsystems=[
                    SerialSystem('pt', copy=i, subsystems=[
                            IndVar('xi',copy=i,val=i+1,size=1),
                            SerialSystem('cpl', copy=i, subsystems=[
                                    Var_yi('yi', i),
                                    Var_zi('zi', i),
                                    ])
                            ], NL='NLN_GS')
                    for i in range(2)], NL='NLN_JC'),
        ], NL='NEWTON').setup()

print main.compute().array

if main(['yi',0]).comm is not None:
    print 'yi:', main(['yi',0]).check_derivatives(main.variables.keys())
if main(['zi',0]).comm is not None:
    print 'zi:', main(['zi',0]).check_derivatives(main.variables.keys())

h = 0.1
v0 = numpy.array(main.compute(False).array)
d = numpy.array(main.compute_derivatives('fwd', 'x0', output=False).array)
main('x0').value += h
v = numpy.array(main.compute(False).array)

print (v-v0)/h, d
