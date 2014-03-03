from __future__ import division
from system import *
from mpi4py import MPI
import numpy

class VarA(ImplicitSystem):

    def _declare_global(self):
        return 'a', 1

    def _declare_local(self):
        self._declare_local_size(1)
        self._declare_local_argument('b')

    def _apply_F(self):
        a = self.uVec()[0]
        b = self.pVec('b')[0]
        self.fVec()[0] = b - a**3
        #self.fVec()[0] = 2*a + 3*b - 5

    def _apply_dFdpu(self, mode, arguments):
        a = self.uVec()[0]
        b = self.pVec('b')[0]
        da = self.duVec()[0]
        db = self.dpVec('b')[0]
        dFda = -3*a**2
        dFdb = 1
        #dFda = 2
        #dFdb = 3
        self.dfVec()[0] += dFda * da + dFdb * db

class VarB(ImplicitSystem):

    def _declare_global(self):
        return 'b', 1

    def _declare_local(self):
        self._declare_local_size(1)
        self._declare_local_argument('a')

    def _apply_F(self):
        a = self.pVec('a')[0]
        b = self.uVec()[0]
        self.fVec()[0] = b - numpy.exp(-a)
        #self.fVec()[0] = a + 2*b - 2

    def _apply_dFdpu(self, mode, arguments):
        a = self.pVec('a')[0]
        b = self.uVec()[0]
        da = self.dpVec('a')[0]
        db = self.duVec()[0]
        dFda = numpy.exp(-a)
        dFdb = 1
        #dFda = 1
        #dFdb = 2
        self.dfVec()[0] += dFda * da + dFdb * db

main = ParallelSystem('main',[
        VarA(), 
        VarB(),
        ], linSol='GMRES', nlSol='Newton', preCon='None')
      
main.setup() 
main.uVec.array[:] = 1.0
main._solve_F() 
print main.rank, main.uVec.array
