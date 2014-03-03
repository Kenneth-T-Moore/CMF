from __future__ import division
from system import *
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print "size: ",size
print "rank: ",rank


class Var_yi(ExplicitSystem):
    
    def _declare_global(self):
        return 'yi', 1

    def _declare_local(self):
        self._declare_local_size(1)
        self._declare_local_argument('x0')
        self._declare_local_argument(['xi',-1])
        self._declare_local_argument(['ai',-1])

    def _apply_G(self):
        x0 = self.pVec('x0')[0]
        xi = self.pVec(['xi',-1])[0]
        ai = self.pVec(['ai',-1])[0]
        self.uVec()[0] = xi*ai - x0*xi**2+1

    def _apply_dGdp(self, mode, arguments):
        x0 = self.pVec('x0')[0]
        xi = self.pVec(['xi',-1])[0]
        ai = self.pVec(['ai',-1])[0]        
        if mode == 'fwd':
            if self._ID('x0') in arguments:
                self.dgVec()[0] += -(xi**2) * self.dpVec('x0')[0]
            if self._ID(['xi',-1]) in arguments:
                self.dgVec()[0] += (ai - 2*x0*xi) * self.dpVec(['xi',-1])[0]
            if self._ID(['ai',-1]) in arguments:
                self.dgVec()[0] += xi * self.dpVec(['ai',-1])[0]
        elif mode == 'rev':
            if self._ID(['x0',0]) in arguments:
                self.dpVec('x0')[0] += -(xi**2) * self.dgVec()[0]
            if self._ID(['xi',-1]) in arguments:
                self.dpVec(['xi',-1])[0] += (ai-2*x0*xi)*self.dgVec()[0]
            if self._ID(['ai',-1]) in arguments:
                self.dpVec(['ai',-1])[0] += xi * self.dgVec()[0]

class Var_ai(ImplicitSystem):
    
    def _declare_global(self):
        return 'ai', 1
    
    def _declare_local(self):
        self._declare_local_size(1)
        self._declare_local_argument('x0')
        self._declare_local_argument(['xi',-1])
        self._declare_local_argument(['yi',-1])

    def _apply_F(self):
        x0 = self.pVec('x0')[0]
        xi = self.pVec(['xi',-1])[0]
        yi = self.pVec(['yi',-1])[0]
        ai = self.uVec()[0]
        self.fVec()[0] = numpy.exp(ai) + yi/xi - x0

    def _solve_F(self):
        x0 = self.pVec('x0')[0]
        xi = self.pVec(['xi',-1])[0]
        yi = self.pVec(['yi',-1])[0]
        self.uVec()[0] = numpy.log(abs(x0-yi/xi))

    def _apply_dFdpu(self, mode, arguments):
        x0 = self.pVec('x0')[0]
        xi = self.pVec(['xi',-1])[0]
        yi = self.pVec(['yi',-1])[0]
        ai = self.uVec()[0]        
        if mode == 'fwd':
            if self._ID(['x0', 0]) in arguments:
                self.dfVec()[0] += -self.dpVec('x0')[0]
            if self._ID(['xi', -1]) in arguments:
                self.dfVec()[0] += -(yi/xi**2) * self.dpVec(['xi',-1])[0]
            if self._ID(['yi', -1]) in arguments:
                self.dfVec()[0] += (1/xi) * self.dpVec(['yi',-1])[0]
            if self._ID(['ai', -1]) in arguments:
                self.dfVec()[0] += numpy.exp(self.duVec(['ai',-1])[0])

        elif mode == 'rev':
            if self._ID(['x0', 0]) in arguments:
                self.dpVec(['x0',0])[0] += -self.dfVec()[0]
            if self._ID(['xi',-1]) in arguments:
                self.dpVec(['xi',-1])[0] += -(yi/xi**2) * self.dfVec()[0]
            if self._ID(['yi',-1]) in arguments:
                self.dpVec(['yi',-1])[0] += (1/xi) * self.dfVec()[0]
            if self._ID(['ai',-1]) in arguments:
                self.duVec(['ai',-1])[0] += numpy.exp(self.dfVec()[0])

class Var_bi(ImplicitSystem):
    
    def _declare_global(self):
        return 'bi', 1

    def _declare_local(self):
        self._declare_local_size(1)
        self._declare_local_argument(['xi',-1])
        self._declare_local_argument(['ai',-1])

    def _apply_F(self):
        xi = self.pVec(['xi',-1])[0]
        ai = self.pVec(['ai',-1])[0]
        bi = self.uVec()[0]
        self.fVec()[0] = numpy.log(abs(bi))+bi-ai-xi

    def _solve_F(self):
        xi = self.pVec(['xi',-1])[0]
        ai = self.pVec(['ai',-1])[0]
        bi = self.uVec()[0]
        self.uVec()[0] = ai+xi-numpy.log(abs(bi))

    def _apply_dFdpu(self, mode, arguments):
        xi = self.pVec(['xi',-1])[0]
        ai = self.pVec(['ai',-1])[0]
        bi = self.uVec()[0]

        if mode == 'fwd':
            if self._ID(['xi',-1]) in arguments:
                self.dfVec()[0] += -1 * self.dpVec(['xi',-1])[0]
            if self._ID(['ai',-1]) in arguments:
                self.dfVec()[0] += -1 * self.dpVec(['ai',-1])[0]
            if self._ID(['bi',-1]) in arguments:
                self.dfVec()[0] += (1 + 1/bi) * self.duVec(['bi',-1])[0]
        if mode == 'rev':
            if self._ID(['xi',-1]) in arguments:
                self.dpVec(['xi',-1])[0] += -1 * self.dfVec()[0]
            if self._ID(['ai',-1]) in arguments:
                self.dpVec(['ai',-1])[0] += -1 * self.dfVec()[0]
            if self._ID(['bi',-1]) in arguments:
                self.duVec(['bi',-1])[0] += (1 + 1/bi) * self.dfVec()[0]

main = \
    SerialSystem('main',[
        IndependentSystem('x0',copy=0,value=10.0,size=1),
        SerialSystem('pts',[
                    SerialSystem('pt',[
                            IndependentSystem('xi',copy=i,value=i,size=1),
                            Var_yi(i),
                            SerialSystem('cons',[
                                    Var_ai(i),
                                    Var_bi(i),
                                    ],i)
                            ],i)
                    for i in range(2)]),
        ])#, linSol='GMRES', nlSol='Newton')

main.setup()
main.uVec.array[:] = 2.0
print main.fVec.array
main._solve_F()
print main.fVec.array
#main.subsystems[1].subsystems[0].subsystems[0]._initializeCommunicators(comm)
#if rank == 0:
#    print main.subsystems[1].subsystems[0].subsystems[2].subsystems[1].cVec
#    main.subsystems[1].subsystems[0].subsystems[2].subsystems[1]._evalC()
#    print main.subsystems[1].subsystems[0].subsystems[2].subsystems[1].cVec

