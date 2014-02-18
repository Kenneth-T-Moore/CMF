from __future__ import division
from system import *
from mpi4py import MPI
import numpy
import time


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
        self.value = numpy.array([self.copy+1])


class Var_yi(ExplicitVariable):

    def _declare(self):
        return 'yi', 1

    def _declareArguments(self):
        self._setLocalSize(1)
        self._setArgument('x0')
        self._setArgument('xi', [0], -1)
        #self._setArgument('zi', [0], -1)

    def _solve(self):
        x0 = self.vVec(['x0'])[0]
        xi = self.vVec(['xi',-1])[0]
        zi1 = 2#self.vVec(['zi'])[0]
        self.vVec()[0] = xi*zi1 - x0*xi**2
        print self.vVec()[0], x0, xi, self.vVec


main = \
    SerialSystem('main',[
        Var_x0(), 
        ParallelSystem('pts',[
                    SerialSystem('pt',[
                            Var_xi(i),
                            Var_yi(i),
                            ],i) 
                    for i in range(2)]),
        ])

#print main.variables
main.initialize()

print 'v', MPI.COMM_WORLD.rank, main.vVarPETSc.array
main._evalCinv()
print 'v', MPI.COMM_WORLD.rank, main.vVarPETSc.array

if MPI.COMM_WORLD.rank is 0 or True:
    print 'vVec', main.vVec
    print 'cVec', main.cVec
