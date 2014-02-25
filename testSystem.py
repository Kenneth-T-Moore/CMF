from __future__ import division
from system import *
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print "size: ",size
print "rank: ",rank

class Var_x0(IndependentVariable):
    
    def _declare(self):
        return 'x0', 1

    def _declareArguments(self):
        self._setLocalSize(1)
        self.value = numpy.array([10])

class Var_xi(IndependentVariable):
    
    def _declare(self):
        return 'xi', 1

    def _declareArguments(self):
        self._setLocalSize(1)
        self.value = numpy.array([self.copy])

class Var_yi(ExplicitVariable):
    
    def _declare(self):
        return 'yi', 1

    def _declareArguments(self):
        self._setLocalSize(1)
        self._setArgument('x0')
        self._setArgument('xi', indices=[0], copy=-1)
        self._setArgument('ai', indices=[0], copy=-1)

    def _solve(self):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]
        self.vVec()[0] = xi*ai - x0*xi**2+1

    def _applyJacobian(self, arguments):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]
        
        if ['x0',0] in arguments:
            self.yVec([])[0] += -(xi**2) * self.xVec(['x0'])[0]
        if ['xi',-1] in arguments:
            self.yVec([])[0] += (ai - 2*x0*xi) * self.xVec(['xi',-1])[0]
        if ['ai',-1] in arguments:
            self.yVec([])[0] += xi * self.xVec(['ai',-1])[0]

    def _applyJacobian_T(self, arguments):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]

        if ['x0',0] in arguments:
            self.xVec(['x0'])[0] += -(xi**2) * self.yVec([])[0]
        if ['xi',-1] in arguments:
            self.xVec(['xi',-1])[0] += (ai-2*x0*xi)*self.yVec([])[0]
        if ['ai',-1] in arguments:
            self.xVec(['ai',-1])[0] += xi * self.yVec([])[0]

class Var_ai(ImplicitVariable):
    
    def _declare(self):
        return 'ai', 1
    
    def _declareArguments(self):
        self._setLocalSize(1)
        self._setArgument('x0')
        self._setArgument('xi', indices=[0], copy=-1)
        self._setArgument('yi', indices=[0], copy=-1)

    def _evalC(self):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        yi = self.vVec(['yi',-1])[0]
        ai = self.vVec()[0]
        self.cVec()[0] = numpy.exp(ai) + yi/xi - x0

    def _evalCinv(self):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        yi = self.vVec(['yi',-1])[0]
        self.vVec()[0] = numpy.log(x0-yi/xi)

    def _applyJ(self, mode, arguments):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        yi = self.vVec(['yi',-1])[0]
        ai = self.vVec()[0]
        
        if mode == 'fwd':
            if ['x0', 0] in arguments:
                self.yVec()[0] += -self.xVec(['x0'])[0]
            if ['xi', -1] in arguments:
                self.yVec()[0] += -(yi/xi**2) * self.xVec(['xi',-1])[0]
            if ['yi', -1] in arguments:
                self.yVec()[0] += (1/xi) * self.xVec(['yi',-1])[0]
            if ['ai', -1] in arguments:
                self.yVec()[0] += numpy.exp(self.xVec(['ai',-1])[0])

        elif mode == 'rev':
            if ['x0', 0] in arguments:
                self.xVec(['x0',0])[0] += -self.yVec()[0]
            if ['xi',-1] in arguments:
                self.xVec(['xi',-1])[0] += -(yi/xi**2) * self.yVec()[0]
            if ['yi',-1] in arguments:
                self.xVec(['yi',-1])[0] += (1/xi) * self.yVec()[0]
            if ['ai',-1] in arguments:
                self.xVec(['ai',-1])[0] += numpy.exp(self.yVec()[0])

class Var_bi(ImplicitVariable):
    
    def _declare(self):
        return 'bi', 1

    def _declareArguments(self):
        self._setLocalSize(1)
        self._setArgument('xi',indices=[0],copy=-1)
        self._setArgument('ai',indices=[0],copy=-1)

    def _evalC(self):
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]
        bi = self.vVec()[0]
        self.cVec()[0] = numpy.log(bi)+bi-ai-xi

    def _evalCinv(self):
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]
        bi = self.vVec()[0]
        self.vVec()[0] = ai+xi-numpy.log(bi)

    def _applyJ(self, mode, arguments):
        xi = self.vVec(['xi',-1])[0]
        ai = self.vVec(['ai',-1])[0]
        bi = self.vVec()[0]

        if mode == 'fwd':
            if ['xi',-1] in arguments:
                self.yVec()[0] += -1 * self.xVec(['xi',-1])[0]
            if ['ai',-1] in arguments:
                self.yVec()[0] += -1 * self.xVec(['ai',-1])[0]
            if ['bi',-1] in arguments:
                self.yVec()[0] += (1 + 1/bi) * self.xVec(['bi',-1])[0]
        if mode == 'rev':
            if ['xi',-1] in arguments:
                self.xVec(['xi',-1])[0] += -1 * self.yVec()[0]
            if ['ai',-1] in arguments:
                self.xVec(['ai',-1])[0] += -1 * self.yVec()[0]
            if ['bi',-1] in arguments:
                self.xVec(['bi',-1])[0] += (1 + 1/bi) * self.yVec()[0]

main = \
    SerialSystem('main',[
        Var_x0(),
        ParallelSystem('pts',[
                    SerialSystem('pt',[
                            Var_xi(i),
                            Var_yi(i),
                            SerialSystem('cons',[
                                    Var_ai(i),
                                    Var_bi(i),
                                    ],i)
                            ],i)
                    for i in range(2)]),
        ])

main.initialize()
#main.subsystems[1].subsystems[0].subsystems[0]._initializeCommunicators(comm)
if rank == 0:
    print main.subsystems[1].subsystems[0].subsystems[2].subsystems[1].cVec
    main.subsystems[1].subsystems[0].subsystems[2].subsystems[1]._evalC()
    print main.subsystems[1].subsystems[0].subsystems[2].subsystems[1].cVec

'''
x0 = Var_x0()
xi = []
for i in xrange(0,2):
    xi.append(Var_xi(i))

print xi[0].copy
print xi[1].copy
xi[0]._initializeCommunicators(comm)
xi[1]._initializeCommunicators(comm)
print xi[0].value
print xi[1].value

print xi[0].vVec(['xi',-1])
'''
