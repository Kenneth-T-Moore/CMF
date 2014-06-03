from __future__ import division
from framework import *

class Discipline1(ExplicitSystem):
    """Evaluates the equation
    y1 = z1**2 + z2 + x1 - 0.2*y2"""

    def _declare(self):
        self._declare_variable('y1')
        self._declare_argument('x1')
        self._declare_argument('z1')
        self._declare_argument('z2')
        self._declare_argument('y2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x1, z1, z2, y2 = p('x1')[0], p('z1')[0], p('z2')[0], p('y2')[0]
        y1 = u('y1')
        y1[0] = z1**2 + z2 + x1 - 0.2*y2

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x1, z1, z2, y2 = p('x1')[0], p('z1')[0], p('z2')[0], p('y2')[0]
        y1 = u('y1')[0]
        dx1, dz1, dz2, dy2 = dp('x1'), dp('z1'), dp('z2'), dp('y2')
        dy1 = dg('y1')

        dy1_dx1 = 1.0
        dy1_dz1 = 2.0*dz1
        dy1_dz2 = 1.0
        dy1_dy2 = -0.2*y2

        if self.mode == 'fwd':
            dy1[0] = 0
            if self.get_id('x1') in args:
                dy1[0] += dy1_dx1*dx1[0]
            if self.get_id('z1') in args:
                dy1[0] += dy1_dz1*dz1[0]
            if self.get_id('z2') in args:
                dy1[0] += dy1_dz2*dz2[0]
            if self.get_id('y2') in args:
                dy1[0] += dy1_dy2*dy2[0]
        else:
            if self.get_id('x1') in args:
                dx1[0] = dy1_dx1*dy1[0]
            if self.get_id('z1') in args:
                dz1[0] = dy1_dz1*dy1[0]
            if self.get_id('z2') in args:
                dz2[0] = dy1_dz2*dy1[0]
            if self.get_id('y2') in args:
                dy2[0] = dy1_dy2*dy1[0]


class Discipline2(ExplicitSystem):
    """Evaluates the equation
    y2 = y1**(.5) + z1 + z2"""

    def _declare(self):
        self._declare_variable('y2')
        self._declare_argument('z1')
        self._declare_argument('z2')
        self._declare_argument('y1')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        z1, z2, y1 =  p('z1')[0], p('z2')[0], p('y1')[0]
        y2 = u('y2')
        y2[0] = y1**(.5) + z1 + z2

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        z1, z2, y1 = p('z1')[0], p('z2')[0], p('y1')[0]
        y2 = u('y2')[0]
        dz1, dz2, dy1 = dp('z1'), dp('z2'), dp('y1')
        dy2 = dg('y2')

        dy2_dz1 = 1.0
        dy2_dz2 = 1.0
        dy2_dy1 = 0.5*y2**(-0.5)

        if self.mode == 'fwd':
            dy2[0] = 0
            if self.get_id('z1') in args:
                dy2[0] += dy2_dz1*dz1[0]
            if self.get_id('z2') in args:
                dy2[0] += dy2_dz2*dz2[0]
            if self.get_id('y2') in args:
                dy2[0] += dy2_dy1*dy1[0]
        else:
            if self.get_id('z1') in args:
                dz1[0] = dy2_dz1*dy2[0]
            if self.get_id('z2') in args:
                dz2[0] = dy2_dz2*dy2[0]
            if self.get_id('y2') in args:
                dy1[0] = dy2_dy1*dy2[0]

main = SerialSystem('main', subsystems=[
    IndVar('z1', val=5.0),
    IndVar('z2', val=2.0),
    IndVar('x1', val=1.0),
    Discipline1('y1'),
    Discipline2('y2'),
    ]).setup()


print main.compute()
print 'fwd'
print main.compute_derivatives('fwd', 'z1', output=False)
print main.compute_derivatives('fwd', 'z2', output=False)
print main.compute_derivatives('fwd', 'x1', output=False)
print 'rev'
print main.compute_derivatives('rev', 'z1', output=False)
print main.compute_derivatives('rev', 'z2', output=False)
print main.compute_derivatives('rev', 'x1', output=False)

main.check_derivatives_all(fwd=True)