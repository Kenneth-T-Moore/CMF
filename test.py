from __future__ import division
from component import *
from mpi4py import MPI
import numpy


class Var_x0(SimpleComponent):

    def _declare(self):
        return 'x0', 1

    def _initializeArguments(self):
        self.localSize = 1


class Var_xi(SimpleComponent):

    def _declare(self):
        return 'x', 1

    def _initializeArguments(self):
        self.localSize = 1


class Var_yi(SimpleComponent):

    def _declare(self):
        return 'y', 1

    def _initializeArguments(self):
        self.localSize = 1
        self._addArgument('x0', [0])
        self._addArgument('xi', [0])


class Var_yi(SimpleComponent):

    def _declare(self):
        return 'y', 1

    def _initializeArguments(self):
        self.localSize = 1


main = \
    SerialComponent('main',[
        Var_x0(), 
        ParallelComponent('pts',[
                    SerialComponent('pt',[
                            Var_xi(i),
                            Var_yi(i),
                            ],i) 
                    for i in range(2)]),
        ])
    
print main.variables
main.initialize()
if MPI.COMM_WORLD.rank is 0 or True:
    print 'vVec', main.vVec
    print 'cVec', main.cVec
