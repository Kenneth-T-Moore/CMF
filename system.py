from __future__ import division
from mpi4py import MPI
from petsc4py import PETSc
import numpy
from collections import OrderedDict



class Vec(OrderedDict):
    """ An abstract vector with a dict of views to a PETSc.Vec """

    def __init__(self, system, array):
        super(Vec,self).__init__()
        self.system = system
        self.array = array
        self.PETSc = self._initialize()

class BufferVec(Vec):
    """ A Vec for the parameter vector of a nonlinear system """

    def _initialize(self):
        system = self.system
        i1, i2 = 0, 0
        for n,c in system.variables:
            self[n,c] = OrderedDict()
        for subsystem in system.localSubsystems:
            for (n1, c1) in subsystem.variables:
                args = subsystem.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in subsystem.variables \
                            and (n2, c2) in system.variables:
                        i2 += args[n2,c2].shape[0]
                        self[n1,c1][n2,c2] = self.array[i1:i2]
                        i1 += args[n2,c2].shape[0]
        if system.localSubsystems == []:
            n,c = system.name, system.copy
            args = system.variables[n,c]['args'] 
            if (n,c) in args:
                i2 += args[n,c].shape[0]
                self[n,c][n,c] = self.array[i1:i2]
                i1 += args[n,c].shape[0]                
        return PETSc.Vec().createWithArray(self.array, comm=system.comm)        
    
    def __call__(self, inp1=[], inp2=[]):
        system = self.system
        if inp2 == []:
            return self[system.name, system.copy][system._ID(inp1)]
        else:
            return self[system._ID(inp1)][system._ID(inp2)]

class DataVec(Vec):
    """ A Vec for the unknown vector of a nonlinear system """

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
        system = self.system
        if var == []:
            return self[system.name, system.copy]
        else:
            return self[system._ID(var)]



class System(object):
    """ Nonlinear system base class """

    def _initialize_system(self, name, copy, numReqProcs):
        """ Method called by __init__ to define basic attributes """
        self.name = name
        self.copy = copy
        self.numReqProcs = numReqProcs
        self.variables = OrderedDict()

    def _setup1of6_comms(self):
        """ Receives the communicator and distributes to subsystems """
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._setup1of6_comms_distribute()

        for system in self.localSubsystems:
            system._setup1of6_comms()

    def _setup2of6_sizes(self):
        """ Assembles nProc x nVar array of local variable sizes """
        self.varSizes = numpy.zeros((self.size, len(self.variables)),int)
        index = 0
        for (n,c) in self.variables:
            self.varSizes[self.rank, index] = self.variables[n,c]['size']
            index += 1
        self.comm.Allgather(self.varSizes[self.rank,:],self.varSizes)

        """ Computes sizes of arguments that are parameters for the current system """
        self.argSizes = numpy.zeros(self.size, int)
        for subsystem in self.localSubsystems:
            for (n1, c1) in subsystem.variables:
                args = subsystem.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in subsystem.variables \
                            and (n2, c2) in self.variables:
                        self.argSizes[self.rank] += args[n2,c2].shape[0]
        if self.localSubsystems == []:
            n,c = self.name, self.copy
            args = self.variables[n,c]['args'] 
            if (n,c) in args:
                self.argSizes[self.rank] += args[n,c].shape[0]
        self.comm.Allgather(self.argSizes[self.rank], self.argSizes)

        for system in self.localSubsystems:
            system._setup2of6_sizes()

    def _setup3of6_vecs(self, u, f, du, df):
        """ Creates DataVecs and BufferVecs """
        self.uVec = DataVec(self, u)
        self.fVec = DataVec(self, f)
        self.duVec = DataVec(self, du)
        self.dfVec = DataVec(self, df)

        i1, i2 = 0, 0
        for subsystem in self.localSubsystems:
            i2 += numpy.sum(subsystem.varSizes[subsystem.rank,:])
            subsystem._setup3of6_vecs(u[i1:i2], f[i1:i2], du[i1:i2], df[i1:i2])
            i1 += numpy.sum(subsystem.varSizes[subsystem.rank,:])

        m = self.argSizes[self.rank]
        self.pVec = BufferVec(self, numpy.zeros(m))
        self.dpVec = BufferVec(self, numpy.zeros(m))

        self.dgVec = self.dfVec

    def _setup4of6_propagateBufferVecs(self):
        """ Propagates args in BufferVecs down and up the system hierarchy """
        for subsystem in self.localSubsystems:
            for (n1, c1) in subsystem.variables:
                for (n2, c2) in self.pVec[n1,c1]:
                    subsystem.pVec[n1,c1][n2,c2] = self.pVec[n1,c1][n2,c2]
                    subsystem.dpVec[n1,c1][n2,c2] = self.dpVec[n1,c1][n2,c2]
            subsystem._setup4of6_propagateBufferVecs()
            for (n1, c1) in subsystem.variables:
                for (n2, c2) in subsystem.pVec[n1,c1]:
                    self.pVec[n1,c1][n2,c2] = subsystem.pVec[n1,c1][n2,c2]
                    self.dpVec[n1,c1][n2,c2] = subsystem.dpVec[n1,c1][n2,c2]

    def _setup5of6_scatters(self):
        self._setup5of6_scatters_create()

        for system in self.localSubsystems:
            system._setup5of6_scatters()

    def _setup6of6_solvers(self):
        class Mat(object):
            def __init__(self, op, x, y, mode, args=None):
                self.op = op
                self.x = x
                self.y = y
                self.mode = mode
                self.args = args
        class JacMat(Mat):
            def mult(self, A, x, y):
                self.x.array[:] = x.array[:]
                self.y.array[:] = 0.0
                self.op(self.mode, self.args)
                y.array[:] = self.y.array[:]
        class PCMat(Mat):
            def apply(self, A, x, y):
                self.x.array[:] = x.array[:]
                self.y.array[:] = 0.0
                self.op(self.mode)
                y.array[:] = self.y.array[:]
        class Monitor(object):
            def __init__(self, system):
                self.system = system
            def __call__(self, ksp, its, norm):
                if its == 0:
                    self.norm0 = norm if norm != 0.0 else 1.0
                self.system._print('   GMRES', its, norm/self.norm0)

        m = numpy.sum(self.varSizes[self.rank,:])
        M = numpy.sum(self.varSizes)
        args = self.variables.keys()

        ksp = {}
        for mode in ['fwd']:#, 'rev']:
            if 'preCon' in self.kwargs:
                preCon = self.kwargs['preCon']
            else:
                preCon = self.defaultPreCon

            if preCon == 'None':
                solveJac = self._solve_dFdu_empty
            elif preCon == 'block triangular':
                solveJac = self._solve_dFdu_blockTri
            elif preCon == 'block diagonal':
                solveJac = self._solve_dFdu_blockDiag
            elif preCon == 'block GS':
                solveJac = self._solve_dFdu_GS
            elif preCon == 'block Jacobi':
                solveJac = self._solve_dFdu_Jacobi

            if mode == 'fwd':
                x, y = self.duVec, self.dfVec
            elif mode == 'rev':
                x, y = self.dfVec, self.duVec

            jac = PETSc.Mat().createPython([M,M], comm=self.comm)
            jac.setPythonContext(JacMat(self._apply_dFdpu, x, y, mode, args))
            jac.setUp()

            ksp[mode] = PETSc.KSP().create(comm=self.comm)
            ksp[mode].setOperators(jac)
            ksp[mode].setType('fgmres')
            ksp[mode].setGMRESRestart(10)
            ksp[mode].setPCSide(PETSc.PC.Side.RIGHT)
            ksp[mode].setMonitor(Monitor(self))

            pc = ksp[mode].getPC()
            pc.setType('python')
            pc.setPythonContext(PCMat(solveJac, y, x, mode))

        self.ksp = ksp
        self.sol = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)
        self.rhs = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)

        for system in self.localSubsystems:
            system._setup6of6_solvers()

    def _apply_F(self):
        """ Evaluate function, (p,u) |-> f [overwrite] """
        pass

    def _apply_dFdpu(self, mode, arguments):
        """ Apply Jacobian, (dp,du) |-> df [fwd] or df |-> (dp,du) [rev] [add] """
        pass

    def _solve_F(self):
        """ (Possibly inexact) solve, p |-> u [overwrite] """
        if 'nlSol' in self.kwargs:
            nlSol = self.kwargs['nlSol']
        else:
            nlSol = self.defaultNonlinSol

        if nlSol == 'block triangular':
            self._solve_F_blockTri()
        elif nlSol == 'block diagonal':
            self._solve_F_blockDiag()
        elif nlSol == 'block GS':
            self._solve_F_GS()
        elif nlSol == 'block Jacobi':
            self._solve_F_Jacobi()
        elif nlSol == 'Newton':
            self._solve_F_Newton()

    def _solve_dFdu(self, mode):
        """ Apply Jac. inv., df |-> du [fwd] or du |-> df (rev) [overwrite] """
        if 'linSol' in self.kwargs:
            linSol = self.kwargs['linSol']
        else:
            linSol = self.defaultLinSol

        if linSol == 'block triangular':
            self._solve_dFdu_blockTri(mode)
        elif linSol == 'block diagonal':
            self._solve_dFdu_blockDiag(mode)
        elif linSol == 'block GS':
            self._solve_dFdu_GS(mode)
        elif linSol == 'block Jacobi':
            self._solve_dFdu_Jacobi(mode)
        elif linSol == 'GMRES':
            self._solve_dFdu_GMRES(mode)

    def _solve_F_Newton(self, ilimit=100, atol=1e-12, rtol=1e-6):
        counter = 0
        norm = self._compute_residual()
        norm0 = norm if norm != 0.0 else 1.0
        self._print('Newton', counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self.dfVec.array[:] = -self.fVec.array[:]
            self._solve_dFdu('fwd')
            norm = self._line_search()
            counter += 1
            self._print('Newton', counter, norm/norm0)

    def _line_search(self, ilimit=10, atol=1e-10, rtol=5e-1):
        counter = 0
        alpha = 1.0
        norm0 = self._compute_residual()
        norm0 = norm0 if norm0 != 0.0 else 1.0
        self.uVec.array[:] += alpha * self.duVec.array[:]
        norm = self._compute_residual()
        self._print('   Line search', counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self.uVec.array[:] -= alpha * self.duVec.array[:]
            alpha /= 2.0
            self.uVec.array[:] += alpha * self.duVec.array[:]
            norm = self._compute_residual()
            counter += 1
            self._print('   Line search', counter, norm/norm0)
        return norm

    def _solve_dFdu_GMRES(self, mode):
        self._linear_initialize(mode)
        self.ksp[mode].solve(self.rhs, self.sol)
        self._linear_finalize(mode)

    def _solve_dFdu_empty(self, mode):
        if mode == 'fwd':
            self.duVec.array[:] = self.dfVec.array[:]
        elif mode == 'rev':
            self.dfVec.array[:] = self.duVec.array[:]
        else:
            raise Exception('mode type not recognized')

    def _print(self, method, counter, residual):
        if self.rank == 0:
            print self.name, self.copy, method, counter, residual

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
            scatter = subsystem.scatterRev
        else:
            raise Exception('mode type not recognized')

        if not scatter == None:
            if mode == 'fwd':
                scatter.scatter(vec1, vec2, addv = False, mode = False)
            elif mode == 'rev':
                vec1.array[:] = 0.0
                scatter.scatter(vec2, vec1, addv = True, mode = True)

    def _ID(self, inp):
        if not type(inp) == list:
            return inp, 0
        elif len(inp) == 1:
            return inp[0], 0
        elif inp[1] == -1:
            return inp[0], self.copy
        else:
            return inp[0], inp[1]

    def _linear_initialize(self, mode):
        if mode == 'fwd':
            self.rhs.array[:] = self.dfVec.array[:]
            self.sol.array[:] = self.duVec.array[:]
        elif mode == 'rev':
            self.rhs.array[:] = self.duVec.array[:]
            self.sol.array[:] = self.dfVec.array[:]
        else:
            raise Exception('Vec type not recognized') 

    def _linear_finalize(self, mode):
        if mode == 'fwd':
            self.dfVec.array[:] = self.rhs.array[:]
            self.duVec.array[:] = self.sol.array[:]
        elif mode == 'rev':
            self.duVec.array[:] = self.rhs.array[:]
            self.dfVec.array[:] = self.sol.array[:]
        else:
            raise Exception('Vec type not recognized') 

    def _compute_linearResidual(self, mode):
        if mode == 'fwd':
            vec = self.dfVec
        elif mode == 'rev':
            vec = self.duVec
        else:
            raise Exception('mode type not recognized')
        vec.array[:] = -self.rhs.array[:]
        self._apply_dFdpu(mode, self.variables.keys())
        vec.PETSc.assemble()
        return vec.PETSc.norm()

    def _compute_residual(self):
        self._apply_F()
        self.fVec.PETSc.assemble()
        return self.fVec.PETSc.norm()

    def setup(self, comm=MPI.COMM_WORLD):
        """ Top-level initialize method called by user """
        self.comm = comm
        self._setup1of6_comms()
        self._setup2of6_sizes()

        m = numpy.sum(self.varSizes[self.rank,:])
        zeros = numpy.zeros
        self._setup3of6_vecs(zeros(m), zeros(m), zeros(m), zeros(m))
        self._setup4of6_propagateBufferVecs()
        self._setup5of6_scatters()
        self._setup6of6_solvers()



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
        self.defaultPreCon = 'None'
        self.defaultLinSol = 'GMRES'
        self.defaultNonlinSol = 'Newton'

    def _declare_global(self):
        pass

    def _declare_local(self):
        pass

    def _declare_local_size(self, size):
        """ Size of the variable vector on the current proc """
        self.variables[self.name, self.copy]['size'] = size

    def _declare_local_argument(self, ID, indices=[0]):
        """ Helper method for declaring arguments """
        arg = self.variables[self.name, self.copy]['args']
        arg[self._ID(ID)] = numpy.array(indices,'i')

    def _setup1of6_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        self.localSubsystems = []
        self._declare_local()

    def _setup5of6_scatters_create(self):
        """ Defines a scatter for when a variable is its own argument """
        n,c = self.name, self.copy
        args = self.variables[n,c]['args']
        if (n,c) in args:
            varIndices = args[n,c]
            m1 = numpy.sum(self.argSizes[:self.rank])
            m2 = numpy.sum(self.argSizes[:self.rank+1])
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
        self._apply_G() # p |-> u

    def _apply_dFdpu(self, mode, arguments):
        if mode == 'fwd':
            self.dfVec.array[:] += self.duVec.array[:]
        elif mode == 'rev':
            self.duVec.array[:] += self.dfVec.array[:]
        else:
            raise Exception('mode type not recognized')
        self.dgVec.array[:] *= -1.0
        self._apply_dGdp(mode, arguments) # dp |-> dg; dg |-> dp
        self.dgVec.array[:] *= -1.0

    def _solve_F(self):
        """ v_i = V_i(v_{j!=i}) """
        self._apply_G()



class IndependentSystem(SimpleSystem):
    """ Variable given by v_i = v_i^* """

    def __init__(self, name, copy=0, value=0, size=1, **kwargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        numReqProcs=1
        self.value = value
        if type(value) == numpy.ndarray:
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

    def _solve_dFdu(self, mode):
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

    def __call__(self, ID):
        return self.allsystems[self._ID(ID)]

    def _setup5of6_scatters_create(self):
        """ First, defines the PETSc Vec application ordering objects """
        getLinspace = lambda m1, m2: numpy.array(numpy.linspace(m1,m2-1,m2-m1),'i')
        varSizes = self.varSizes
        argSizes = self.argSizes

        appIndices = []
        i = self.rank
        for j in xrange(len(self.variables)):
            m1 = numpy.sum(varSizes[:,:j]) + numpy.sum(varSizes[:i,j])
            m2 = m1 + varSizes[i,j]
            appIndices.append(getLinspace(m1,m2))
        appIndices = numpy.concatenate(appIndices)

        m1 = numpy.sum(varSizes[:self.rank,:])
        m2 = numpy.sum(varSizes[:self.rank+1,:])
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
            if ISvar.getSize() == 0:
                return None
            else:
                return PETSc.Scatter().create(self.uVec.PETSc, ISvar,
                                              self.pVec.PETSc, ISarg)

        variableIndex = self.variables.keys().index
        varIndsFull = []
        argIndsFull = []
        i1, i2 = numpy.sum(argSizes[:self.rank]), numpy.sum(argSizes[:self.rank])
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
                        j = variableIndex((n2,c2))
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
        self.comm.Barrier()
        self.scatterFull = createScatter(self, varIndsFull, argIndsFull)

    def _apply_F(self):
        """ Delegate to subsystems """
        self._scatter('u', 'fwd')
        for subsystem in self.localSubsystems:
            subsystem._apply_F()

    def _apply_dFdpu(self, mode, arguments):
        """ Delegate to subsystems """
        self._scatter('du', mode) if mode == 'fwd' else None
        for subsystem in self.localSubsystems:
            subsystem._apply_dFdpu(mode, arguments)
        self._scatter('du', mode) if mode == 'rev' else None



class ParallelSystem(CompoundSystem):

    def _setup1of6_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        self.defaultPreCon = 'block Jacobi'
        self.defaultLinSol = 'block Jacobi'
        self.defaultNonlinSol = 'block Jacobi'

        if len(self.subsystems) > self.size:
            raise Exception("Not enough procs to split comm in " + self.name)

        nSubs = len(self.subsystems)
        assignedProcs = numpy.ones(nSubs, int)
        pctgProcs = numpy.zeros(nSubs)
        targetPctgProcs = [subsystem.numReqProcs for subsystem in self.subsystems]
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

    def _solve_F_Jacobi(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Jacobi """
        counter = 0
        norm = self._compute_residual()
        norm0 = norm if norm != 0.0 else 1.0
        self._print('NL Jacobi', counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_F_blockDiag()
            norm = self._compute_residual()
            counter += 1
            self._print('NL Jacobi', counter, norm/norm0)

    def _solve_F_blockDiag(self):
        """ Solve each subsystem in parallel """
        self._scatter('u', 'fwd')
        for subsystem in self.localSubsystems:
            subsystem._solve_F()

    def _solve_dFdu_Jacobi(self, mode, ilimit=10, atol=1e-6, rtol=1e-4):
        """ Linear block Jacobi """
        self._linear_initialize(mode)
        counter = 0
        norm = self._compute_linearResidual(mode)
        norm0 = norm if norm != 0.0 else 1.0
        self._print('Lin Jacobi', counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_dFdu_blockDiag(mode)
            norm = self._compute_linearResidual(mode)
            counter += 1
            self._print('Lin Jacobi', counter, norm/norm0)
        #self._linear_finalize(mode)

    def _solve_dFdu_blockDiag(self, mode):
        """ Invert each subsystem's block; block diagonal preconditioner """
        self._scatter('du', mode) if mode == 'fwd' else None
        for subsystem in self.localSubsystems:
            subsystem._solve_dFdu(mode)
        self._scatter('du', mode) if mode == 'rev' else None



class SerialSystem(CompoundSystem):

    def _setup1of6_comms_distribute(self):
        """ Defines subsystems on current proc and assigns them comms """
        self.defaultPreCon = 'block GS'
        self.defaultLinSol = 'block GS'
        self.defaultNonlinSol = 'block GS'

        self.localSubsystems = self.subsystems

        for system in self.localSubsystems:
            system.comm = self.comm

    def _solve_F_GS(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Gauss Seidel """
        counter = 0
        norm = self._compute_residual()
        norm0 = norm if norm != 0.0 else 1.0
        self._print('NL GS', counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_F_blockTri()
            norm = self._compute_residual()
            print '*', self.rank, self.fVec.array, self.uVec.array, self.pVec.array
            counter += 1
            self._print('NL GS', counter, norm/norm0)

    def _solve_F_blockTri(self):
        """ Solve each subsystem sequentially """
        for subsystem in self.subsystems:
            self._scatter('u', 'fwd', subsystem)
            subsystem._solve_F()

    def _solve_dFdu_GS(self, mode, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Linear block Gauss Seidel """
        counter = 0
        norm0 = self._compute_linearResidual(mode)
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._solve_dFdu_blockTri(mode)
            norm = self._compute_linearResidual(mode)
            counter += 1

    def _solve_dFdu_blockTri(self, mode):
        """ Block fwd or rev substitution; block triangular preconditioner """
        if mode == 'fwd':
            args = []
            for subsystem in self.subsystems:
                self._scatter('du', mode, subsystem)
                subsystem.dfVec.array[:] *= -1.0
                subsystem._apply_dFdpu(mode, args)
                subsystem.dfVec.array[:] *= -1.0
                subsystem._solve_dFdu(mode)
                args.extend(subsystem.variables)
        elif mode == 'rev':
            self.subsystems.reverse()
            args = [v for v in self.variables]
            for subsystem in self.subsystems:
                for v in xrange(len(subsystem.variables)):
                    args.pop()
                subsystem.duVec.array[:] *= -1.0
                self._scatter('du', mode, subsystem)
                subsystem.duVec.array[:] *= -1.0
                subsystem._solve_dFdu(mode)
                subsystem._apply_dFdpu(mode, args)
            self.subsystems.reverse()
        else:
            raise Exception('Vec type not recognized')
