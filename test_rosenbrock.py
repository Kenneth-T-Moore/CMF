from __future__ import division
from framework import *
from optimization import *


class VarF(ExplicitSystem):

    def _declare(self):
        self._declare_variable('f')
        self._declare_argument('x')
        self._declare_argument('y')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x, y, f = p('x'), p('y'), u('f')
        f[0] = (1-x[0])**2 + 100*(y[0]-x[0]*x[0])**2

    def apply_dGdp(self, args):
        vec = self.vec
        p, dp, du, dg = vec['p'], vec['dp'], vec['du'], vec['dg']
        x, y = p('x'), p('y')
        dx, dy, df = dp('x'), dp('y'), dg('f')

        dfdx = -2*(1-x[0]) - 400*x[0]*(y[0] - x[0]*x[0])
        dfdy = 200*(y[0] - x[0]**2)
        if self.mode == 'fwd':
            df[0] = 0
            if self.get_id('x') in args:
                df[0] += dfdx*dx[0]
            if self.get_id('y') in args:
                df[0] += dfdy*dy[0]
        else:
            if self.get_id('x') in args:
                dx[0] = dfdx*df[0]
            if self.get_id('y') in args:
                dy[0] = dfdy*df[0]
            

main = SerialSystem('main', subsystems=[
        IndVar('x', val=-1.0),
        IndVar('y', val=-1.0),
        VarF('f'),
        ], NL='NLN_GS', LN='LIN_GS').setup()

#print main.compute(False).array
#print main.compute_derivatives('fwd', 'x')
#print main.compute_derivatives('rev', 'f')

opt = Optimization(main)
opt.add_design_variable('x')
opt.add_design_variable('y')
opt.add_objective('f')
opt('SNOPT')
