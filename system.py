from __future__ import division
from mpi4py import MPI
from petsc4py import PETSc
import numpy
from collections import OrderedDict



class OrderedDict0(OrderedDict):
    """ An OrderedDict that can infer copy # when not specified """

    def __init__(self, name, copy, system, array):
        self.name = name
        self.copy = copy
        self.system = system
        self.array = array
        self.isSimple = system.subsystems is []
        super(OrderedDict0,self).__init__()
        self.PETSc = self._initialize()

    def _ID(self, inp):
        if type(inp) is not list or len(inp) == 1:
            return inp[0], 0
        elif inp[1] == -1:
            return inp[0], self.copy
        else:
            return inp[0], inp[1]

class BufferVec(OrderedDict0):
    """ Domain vecs: assumes [n,c][n,c] when arg not specified """

    def _initialize(self):
        system = self.system
        i1, i2 = 0, 0
        for subsystem in system.localSubsystems:
            for (n1, c1) in subsystem.variables:
                self[n1,c1] = OrderedDict()
                args = subsystem.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in subsystem.variables \
                            and (n2, c2) in system.variables:
                        i2 += args[n2,c2].shape[0]
                        self[n1,c1][n2,c2] = self.array[i1:i2]
                        i1 += args[n2,c2].shape[0]
        if system.localSubsystems == []:
            n,c = self.name, self.copy
            args = system.variables[n,c]['args'] 
            if (n,c) in args:
                i2 += args[n,c].shape[0]
                self[n,c][n,c] = self.array[i1:i2]
                i1 += args[n,c].shape[0]
                
        return PETSc.Vec().createWithArray(self.array, comm=system.comm)
        
    
    def __call__(self, inp1=[], inp2=[]):
        if inp2 is []:
            return self[self.name, self.copy][self._ID(inp1)]
        else:
            return self[self._ID(inp1)][self._ID(inp2)]

class DataVec(OrderedDict0):
    """ Codomain vecs: simply wraps the _ID method as a __call__ """

    def _initialize(self): 
        system = self.system       
        i1, i2 = 0, 0
        for i in xrange(len(system.variables)):
            n,c = system.variables.keys()[i]
            i2 += system.varSizes[system.rank,i]
            self[n,c] = self.array[i1:i2]
            i1 += system.varSizes[system.rank,i]

        return PETSc.Vec().createWithArray(self.array, comm=system.comm)

    def __call__(self, var=[]):
        if var is []:
            return self[self.name, self.copy]
        else:
            return self[self._ID(var)]



class System(object):
    """ Nonlinear system base class """

    def _initialize_system(self, name, copy, numReqProcs):
        """ Method called by __init__ to define basic attributes """
        self.name = name
        self.copy = copy
        self.numReqProcs = numReqProcs
        self.variables = OrderedDict()

    def _setup1of5_comms(self):
        """ Receives the communicator and distributes to subsystems """
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._setup1of5_comms_distribute()

        for system in self.localSubsystems:
            system._setup1of5_comms()

    def _setup2of5_sizes(self):
        """ Assembles nProc x nVar array of local variable sizes """
        self.varSizes = numpy.zeros((self.size, len(self.variables)),int)
        index = 0
        for (n,c) in self.variables:
            self.varSizes[self.rank, index] = self.variables[n,c]['size']
            index += 1
        self.comm.Allgather(self.varSizes[self.rank,:],self.varSizes)

        """ Computes sizes of arguments that are parameters for the current system """
        self.argSizes = numpy.zeros(self.size, int)
        for system in self.localSubsystems:
            for (n1, c1) in system.variables:
                args = system.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in system.variables \
                            and (n2, c2) in self.variables:
                        self.argSizes[self.rank] += args[n2,c2].shape[0]
        if self.localSubsystems == []:
            n,c = self.name, self.copy
            args = self.variables[n,c]['args'] 
            if (n,c) in args:
                self.argSizes[self.rank] = args[n,c].shape[0]
        self.comm.Allgather(self.argSizes[self.rank], self.argSizes)

        for system in self.localSubsystems:
            system._setup2of5_sizes()

    def _setup3of5_vecs(self, u, f, du, df):
        """ Creates PETSc Vecs with preallocated (Var) or new (Arg) arrays """
        self.uVec = DataVec(self.name, self.copy, self, u)
        self.fVec = DataVec(self.name, self.copy, self, f)
        self.duVec = DataVec(self.name, self.copy, self, du)
        self.dfVec = DataVec(self.name, self.copy, self, df)

        i1, i2 = 0, 0
        for subsystem in self.localSubsystems:
            i2 += numpy.sum(subsystem.varSizes[subsystem.rank,:])
            subsystem._setup3of5_vecs(u[i1:i2], f[i1:i2], du[i1:i2], df[i1:i2])
            i1 += numpy.sum(subsystem.varSizes[subsystem.rank,:])

        m = self.argSizes[self.rank]
        self.pVec = BufferVec(self.name, self.copy, self, numpy.zeros(m))
        self.dpVec = BufferVec(self.name, self.copy, self, numpy.zeros(m))
        self.dgVec = self.dpVec

    def _setup4of5_propagateBufferVecs(self):
        """ Creates the mapping between the Vec OrderedDicts and data """
        for subsystem in self.localSubsystems:
            for (n1, c1) in subsystem.variables:
                for (n2, c2) in self.pVec[n1,c1]:
                    subsystem.pVec[n1,c1][n2,c2] = self.pVec[n1,c1][n2,c2]
                    subsystem.dpVec[n1,c1][n2,c2] = self.dpVec[n1,c1][n2,c2]
            subsystem._setup4of5_propagateBufferVecs()
            for (n1, c1) in subsystem.variables:
                for (n2, c2) in subsystem.pVec[n1,c1]:
                    self.pVec[n1,c1][n2,c2] = subsystem.pVec[n1,c1][n2,c2]
                    self.dpVec[n1,c1][n2,c2] = subsystem.dpVec[n1,c1][n2,c2]

    def _setup5of5_scatters(self):
        self._setup5of5_scatters_create()

        for system in self.localSubsystems:
            system._setup5of5_scatters()

    def _apply_F(self):
        """ Evaluate function, (u,p) |-> f [overwrite] """
        pass

    def _apply_dFdpu(self, mode, arguments):
        """ Apply Jacobian, (du,dp) |-> df [fwd] or df |-> (du,dp) [rev] [add] """
        pass

    def _solve_F(self):
        """ (Possibly inexact) solve, p |-> u [overwrite] """
        pass

    def _solve_dFdu(self, mode):
        """ Apply Jac. inv., df |-> du [fwd] or du |-> df (rev) [overwrite] """
        pass

    def _scatter(self, vec, mode, subsystem=None):
        """ Perform partial or full scatter """
        if vec == 'u':
            vec1, vec2 = self.uVec.PETSc, self.pVec.PETSc
        elif vec == 'du':
            vec1, vec2 = self.duVec.PETSc, self.dpVec.PETSc
        else:
            raise Exception('vec type not recognized')

        if subsystem == None:
            scatter = self.scatterFull
        elif mode == 'fwd':
            scatter = subsystem.scatterFwd
        elif mode == 'rev':
            vec1, vec2 = vec2, vec1
            scatter = subsystem.scatterRev
        else:
            raise Exception('mode type not recognized')

        if not scatter == None:
            scatter.scatter(vec1, vec2, addv = True, mode = mode == 'rev')

    def setup(self, comm=MPI.COMM_WORLD):
        """ Top-level initialize method called by user """
        self.comm = comm
        self._setup1of5_comms()
        self._setup2of5_sizes()

        m = numpy.sum(self.varSizes[self.rank,:])
        zeros = numpy.zeros
        self._setup3of5_vecs(zeros(m), zeros(m), zeros(m), zeros(m))
        self._setup4of5_propagateBufferVecs()
        self._setup5of5_scatters()



class SimpleSystem(System):
    """ Nonlinear system with only one variable """

    def __init__(self, copy=0, **kwargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        self.kwargs = kwargs
        self.subsystems = []
        name, numReqProcs = self._declare_global()
        self._initialize_system(name, copy, numReqProcs)
        self.variables[name, copy] = {'size':0, 'args':OrderedDict()}
        self.allsystems = []

    def _declare_global(self):
        pass

    def _declare_local(self):
        pass

    def _declare_local_size(self, size):
        """ Size of the variable vector on the current proc """
        self.variables[self.name, self.copy]['size'] = size

    def _declare_local_argument(self, name, copy=0, indices=[0]):
        """ Helper method for declaring arguments """
        copy = self.copy if copy == -1 else copy
        arg = self.variables[self.name, self.copy]['args']
        arg[name,copy] = numpy.array(indices,'i')

    def _setup1of5_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        self.localSubsystems = []
        self._declare_local()

    def _setup5of5_scatters_create(self):
        """ Defines a scatter for when a variable is its own argument """
        n,c = self.name, self.copy
        args = self.variables[n,c]['args']
        if (n,c) in args:
            varIndices = args[n,c]
            m1 = numpy.sum(self.argSizes[:self.rank])
            m2 = m1 + args[n,c].shape[0]
            argIndices = numpy.array(numpy.linspace(m1, m2-1, m2-m1), 'i')
            ISvar = PETSc.IS().createGeneral(varIndices, comm=self.comm)
            ISarg = PETSc.IS().createGeneral(argIndices, comm=self.comm)
            self.scatterFull = PETSc.Scatter().create(self.uVec.PETSc, ISvar,
                                                      self.pVec.PETSc, ISarg)
        else:
            self.scatterFull = None



class ImplicitSystem(SimpleSystem):
    """ Variable implicitly defined by v_i : C_i(v) = 0 """
    pass



class ExplicitSystem(SimpleSystem):
    """ Variable explicitly defined by v_i : V_i(v_{j!=i}) """

    def _apply_F(self):
        """ F_i(p_i,u_i) = u_i - G_i(p_i) = 0 """
        self.fVec.array[:] = 0.0
        self._apply_G()

    def _apply_dFdpu(self, mode, arguments):
        if mode == 'fwd':
            self.dfVec.array[:] += self.duVec.array[:]
        elif mode == 'rev':
            self.duVec.array[:] += self.dfVec.array[:]
        else:
            raise Exception('mode type not recognized')
        self.dfVec.array[:] *= -1.0
        self._apply_dGdp(mode, arguments)
        self.dfVec.array[:] *= -1.0

    def _solve_F(self):
        """ v_i = V_i(v_{j!=i}) """
        self._apply_G()



class IndependentSystem(SimpleSystem):
    """ Variable given by v_i = v_i^* """

    def __init__(self, name, copy=0, value=0, size=1, **kwargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        numReqProcs=1
        self.value = value
        if type(value) is numpy.ndarray:
            size = value.shape[0]

        self.kwargs = kwargs
        self.subsystems = []
        self._initialize_system(name, copy, numReqProcs)
        self.variables[name, copy] = {'size':size, 'args':OrderedDict()}
        self.allsystems = []

    def _apply_F(self):
        """ F_i(p_i,u_i) = u_i - u_i^* = 0 """
        self.fVec.array[:] = 0.0
        self.uVec.array[:] = self.value

    def _apply_dFdpu(self, mode, arguments):
        if mode == 'fwd':
            self.dfVec.array[:] += self.duVec.array[:]
        elif mode == 'rev':
            self.duVec.array[:] += self.dfVec.array[:]
        else:
            raise Exception('mode type not recognized')

    def _solve_F(self):
        self.uVec.array[:] = self.value

    def _applyJinv(self, mode):
        if mode == 'fwd':
            self.duVec.array[:] += self.dfVec.array[:]
        elif mode == 'rev':
            self.dfVec.array[:] += self.duVec.array[:]
        else:
            raise Exception('mode type not recognized')



class CompoundSystem(System):
    """ Nonlinear system with multiple variables; concatenation of subsystems """

    def __init__(self, name, subsystems, copy=0, **kwargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        self.kwargs = kwargs
        self.subsystems = subsystems
        numReqProcs = numpy.sum([subsystem.numReqProcs for subsystem in subsystems])
        self._initialize_system(name, copy, numReqProcs)

        for subsystem in subsystems:
            for n,c in subsystem.variables:
                self.variables[n,c] = subsystem.variables[n,c]

        self.allsystems = []
        for system in subsystems:
            self.allsystems.append(system)
            self.allsystems.extend(system.allsystems)

    def __call__(self, name, copy=0):
        copy = self.copy if copy == -1 else copy
        return self.allsystems[name, copy]

    def _setup5of5_scatters_create(self):
        """ First, defines the PETSc Vec application ordering objects """
        getLinspace = lambda m1, m2: numpy.array(numpy.linspace(m1,m2-1,m2-m1),'i')
        varSizes = self.varSizes

        appIndices = []
        i = self.rank
        for j in xrange(len(self.variables)):
            m1 = numpy.sum(varSizes[:,:j]) + numpy.sum(varSizes[:i,j])
            m2 = m1 + varSizes[i,j]
            appIndices.append(getLinspace(m1,m2))
        appIndices = numpy.concatenate(appIndices)

        m1 = numpy.sum(varSizes[:self.rank,:])
        m2 = m1 + numpy.sum(varSizes[self.rank,:])
        petscIndices = getLinspace(m1,m2)

        ISapp = PETSc.IS().createGeneral(appIndices, comm=self.comm)
        ISpetsc = PETSc.IS().createGeneral(petscIndices, comm=self.comm)
        self.AOvarPETSc = PETSc.AO().createBasic(ISapp, ISpetsc, comm=self.comm)

        """ Next, the scatters are defined """
        def createScatter(self, varInds, argInds):
            merge = lambda x: numpy.concatenate(x) if len(x) > 0 else []
            ISvar = PETSc.IS().createGeneral(merge(varInds), comm=self.comm)
            ISarg = PETSc.IS().createGeneral(merge(argInds), comm=self.comm)
            ISvar = self.AOvarPETSc.app2petsc(ISvar)
            if ISvar.array.shape[0] == 0:
                return None
            else:
                return PETSc.Scatter().create(self.uVec.PETSc, ISvar, self.pVec.PETSc, ISarg)

        variableIndex = self.variables.keys().index
        varIndsFull = []
        argIndsFull = []
        i1, i2 = 0, 0
        for subsystem in self.subsystems:
            varIndsFwd = []
            argIndsFwd = []
            varIndsRev = []
            argIndsRev = []
            for (n1, c1) in subsystem.variables:
                args = subsystem.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in subsystem.variables \
                            and (n2, c2) in self.variables:
                        j = variableIndex((n1,c1))
                        i2 += args[n2,c2].shape[0]
                        varInds = numpy.sum(varSizes[:,:j]) + args[n2,c2]
                        argInds = getLinspace(i1, i2)
                        i1 += args[n2,c2].shape[0]
                        if variableIndex((n1,c1)) > variableIndex((n2,c2)):
                            varIndsFwd.append(varInds)
                            argIndsFwd.append(argInds)
                        else:
                            varIndsRev.append(varInds)
                            argIndsRev.append(argInds)
                        varIndsFull.append(varInds)
                        argIndsFull.append(argInds)
            subsystem.scatterFwd = createScatter(self, varIndsFwd, argIndsFwd)
            subsystem.scatterRev = createScatter(self, varIndsRev, argIndsRev)
        self.scatterFull = createScatter(self, varIndsFull, argIndsFull)

    def _applyF(self):
        """ Delegate to subsystems """
        for subsystem in self.localSubsystems:
            self._scatter('u', 'fwd', subsystem)
            subsystem._applyF()

    def _apply_dFdpu(self, mode, arguments):
        """ Delegate to subsystems """
        for subsystem in self.localSubsystems:
            subsystem._apply_dFdpu(mode, arguments)

    def _computeLinearResidual(self, rhs, mode):
        if mode == 'fwd':
            vec = self.dfVec
        elif mode == 'rev':
            vec = self.duVec
        else:
            raise Exception('mode type not recognized')
        self.dpVec.array[:] = 0.0
        vec.array[:] = -rhs[:]
        self._scatter('du', mode) if mode == 'fwd' else None
        self.apply_dFdpu(mode, self.variables.keys())
        self._scatter('du', mode) if mode == 'rev' else None
        vec.PETSc.assemble()
        return vec.PETSc.norm()



class ParallelSystem(CompoundSystem):

    def _setup1of5_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        if len(self.subsystems) > self.size:
            raise Exception("Not enough procs to split comm in " + self.name)

        nSubs = len(self.subsystems)
        assignedProcs = numpy.ones(nSubs, int)
        pctgProcs = numpy.zeros(nSubs)
        targetPctgProcs = [system.numReqProcs for system in self.subsystems]
        targetPctgProcs = numpy.array(targetPctgProcs, float) / self.numReqProcs

        for i in xrange(self.size - nSubs):
            pctgProcs[:] = assignedProcs/numpy.sum(assignedProcs)
            assignedProcs[numpy.argmax(targetPctgProcs - pctgProcs)] += 1

        color = numpy.zeros(self.size, int)
        index = 0
        for i in xrange(nSubs):
            color[index:index + assignedProcs[i]] = i
            index += assignedProcs[i]

        self.assignedProcs = assignedProcs
        subcomm = self.comm.Split(color[self.rank])
        self.localSubsystems = [self.subsystems[color[self.rank]]]

        for subsystem in self.localSubsystems:
            subsystem.comm = subcomm

    def _solve_F(self):
        """ Jacobi by default """
        self._solve_F_Jacobi()

    def _solve_F_blockDiagSolve(self):
        """ Solve each subsystem in parallel """
        self._scatter('u', 'fwd')
        for subsystem in self.localSubsystems:
            subsystem._solve_F()

    def _solve_F_Jacobi(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Jacobi """
        counter = 0
        self._apply_F()
        self.fVec.PETSc.assemble()
        norm0 = self.fVec.PETSc.norm()
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_F_blockDiagSolve()
            self._apply_F()
            self.fVec.PETSc.assemble()
            norm = self.fVec.PETSc.norm()
            counter += 1

    def _solve_dFdu(self, mode):
        """ Jacobi by default """
        self._solve_dFdu_Jacobi(mode)

    def _solve_dFdu_blockDiagSolve(self, mode):
        """ Invert each subsystem's block; block diagonal preconditioner """
        for subsystem in self.localSubsystems:
            subsystem._applyJinv(mode)

    def _solve_dFdu_Jacobi(self, mode, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Linear block Jacobi """
        if mode == 'fwd':
            rhs = numpy.array(self.dfVec.array)
        elif mode == 'rev':
            rhs = numpy.array(self.duVec.array)
        else:
            raise Exception('Vec type not recognized')

        counter = 0
        norm0 = self.computeLinearResidual(self, rhs, mode)
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_dFdu_blockDiagSolve(mode)
            norm = self.computeLinearResidual(self, rhs, mode)
            counter += 1



class SerialSystem(CompoundSystem):

    def _setup1of5_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        self.localSubsystems = self.subsystems

        for system in self.localSubsystems:
            system.comm = self.comm

    def _solve_F(self):
        """ Gauss Seidel by default """
        self._solve_F_GS()

    def _solve_F_FwdBlockSolve(self):
        """ Solve each subsystem sequentially """
        self._scatter('u', 'fwd')
        for subsystem in self.localSubsystems:
            self._scatter('u', 'fwd', subsystem)
            subsystem._solve_F()

    def _solve_F_GS(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Gauss Seidel """
        counter = 0
        self._apply_F()
        self.fVec.PETSc.assemble()
        norm0 = self.fVec.PETSc.norm()
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            print 'GS', counter, norm
            self._solve_F_FwdBlockSolve()
            self._apply_F()
            self.fVec.PETSc.assemble()
            norm = self.fVec.PETSc.norm()
            counter += 1

    def _solve_dFdu(self, mode):
        """ Jacobi by default """
        self._solve_dFdu_GS(mode)

    def _solve_dFdu_blockTrglSolve(self, mode):
        """ Block fwd or rev substitution; block triangular preconditioner """
        self.dpVec.array[:] = 0.0
        if mode == 'fwd':
            for subsystem in self.localSubsystems:
                self._scatter('du', mode, subsystem)
                subsystem.dfVec.array[:] *= -1.0
                subsystem._apply_dFdpu(mode, self.variables.keys())
                subsystem.dfVec.array[:] *= -1.0
                subsystem._solve_dFdu(mode)
        elif mode == 'rev':
            for subsystem in self.localSubsystems:
                subsystem.duVec.array[:] *= -1.0
                self._scatter('xVec', mode, subsystem)
                subsystem.duVec.array[:] *= -1.0
                subsystem._solve_dFdu(mode)
                subsystem._apply_dFdpu(mode, self.variables.keys())
        else:
            raise Exception('Vec type not recognized')

    def _solve_dFdu_GS(self, mode, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Linear block Gauss Seidel """
        if mode == 'fwd':
            rhs = numpy.array(self.dfVec.array)
        elif mode == 'rev':
            rhs = numpy.array(self.duVec.array)
        else:
            raise Exception('Vec type not recognized')

        counter = 0
        norm0 = self.computeLinearResidual(self, rhs, mode)
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_dFdu_blockTrglSolve(mode)
            norm = self.computeLinearResidual(self, rhs, mode)
            counter += 1
