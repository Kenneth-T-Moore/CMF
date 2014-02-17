from __future__ import division
from mpi4py import MPI
from petsc4py import PETSc
import numpy
from collections import OrderedDict



class System(object):
    """ Nonlinear system base class """

    def _initializeSystem(self, name, copy, numReqProcs, kwdargs):
        """ Method called by __init__ to define basic attributes """
        self.name = name
        self.copy = copy
        self.numReqProcs = numReqProcs
        self.kwdargs = kwdargs
        self.variables = OrderedDict()

        class OrderedDict0(OrderedDict):
            """ An OrderedDict that can infer copy # when not specified """
            def __init__(self, copy):
                self.copy = copy
                super(OrderedDict0,self).__init__()
            def _ID(self, inp):
                if len(inp) == 1:
                    return inp[0], 0
                elif inp[1] == -1:
                    return inp[0], self.copy
                else:
                    return inp[0], inp[1]

        class OrderedDictDomVec(OrderedDict0):
            """ Domain vecs: assumes [n,c][n,c] when arg not specified """
            def __call__(self, var, arg=None):
                if arg == None:
                    return self[self._ID(var)][None,None]
                else:
                    return self[self._ID(var)][self._ID(arg)]

        class OrderedDictCodVec(OrderedDict0):
            """ Codomain vecs: simply wraps the _ID method as a __call__ """
            def __call__(self, var):
                return self[self._ID(var)]

        self.vVec = OrderedDictDomVec(copy)
        self.xVec = OrderedDictDomVec(copy)
        self.cVec = OrderedDictCodVec(copy)
        self.yVec = OrderedDictCodVec(copy)

        self.vVec[name, copy] = OrderedDict()
        self.xVec[name, copy] = OrderedDict()

    def _initializeSizeArrays(self):
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
            system._initializeSizeArrays()

    def _initializePETScVecs(self, v, x, c, y):
        """ Creates PETSc Vecs with preallocated (Var) or new (Arg) arrays """
        self.vVarPETSc = PETSc.Vec().createWithArray(v, comm=self.comm)
        self.xVarPETSc = PETSc.Vec().createWithArray(x, comm=self.comm)
        self.cVarPETSc = PETSc.Vec().createWithArray(c, comm=self.comm)
        self.yVarPETSc = PETSc.Vec().createWithArray(y, comm=self.comm)

        m = self.argSizes[self.rank]
        zeros = numpy.zeros
        self.vArgPETSc = PETSc.Vec().createWithArray(zeros(m), comm=self.comm)
        self.xArgPETSc = PETSc.Vec().createWithArray(zeros(m), comm=self.comm)

        i1, i2 = 0, 0
        for system in self.localSubsystems:
            i2 += numpy.sum(system.varSizes[self.rank,:])
            system._initializePETScVecs(v[i1:i2], x[i1:i2], c[i1:i2], y[i1:i2])
            i1 += numpy.sum(system.varSizes[self.rank,:])

    def _initializeVecs(self):
        """ Creates the mapping between the Vec OrderedDicts and data """
        i1, i2 = 0, 0
        for i in xrange(len(self.variables)):
            n,c = self.variables.keys()[i]
            i2 += self.varSizes[self.rank,i]
            self.vVec[n,c][None,None] = self.vVarPETSc.array[i1:i2]
            self.xVec[n,c][None,None] = self.xVarPETSc.array[i1:i2]
            self.cVec[n,c] = self.cVarPETSc.array[i1:i2]
            self.yVec[n,c] = self.yVarPETSc.array[i1:i2]
            i1 += self.varSizes[self.rank,i]

        i1, i2 = 0, 0
        for system in self.localSubsystems:
            for (n1, c1) in system.variables:
                args = system.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in system.variables \
                            and (n2, c2) in self.variables:
                        i2 += args[n2,c2].shape[0]
                        self.vVec[n1,c1][n2,c2] = self.vArgPETSc.array[i1:i2]
                        self.xVec[n1,c1][n2,c2] = self.xArgPETSc.array[i1:i2]
                        i1 += args[n2,c2].shape[0]

            for (n1, c1) in system.variables:
                for (n2, c2) in self.vVec[n1,c1]:
                    system.vVec[n1,c1][n2,c2] = self.vVec[n1,c1][n2,c2]
                    system.xVec[n1,c1][n2,c2] = self.xVec[n1,c1][n2,c2]
            system._initializeVecs()
            for (n1, c1) in system.variables:
                for (n2, c2) in system.vVec[n1,c1]:
                    self.vVec[n1,c1][n2,c2] = system.vVec[n1,c1][n2,c2]
                    self.xVec[n1,c1][n2,c2] = system.xVec[n1,c1][n2,c2]

    def _evalC(self):
        """ Evaluate constraints, vVec -> cVec [overwrite] """
        pass

    def _applyJ(self, mode, arguments):
        """ Apply Jacobian, xVec -> yVec (fwd) or yVec -> xVec (rev) [add] """
        pass

    def _evalCinv(self):
        """ (Possibly inexact) solve, vVec -> vVec [overwrite] """
        pass

    def _applyJinv(self, mode):
        """ Apply Jac. inv., yVec -> xVec (fwd) or xVec -> yVec (rev) [overwrite] """
        pass

    def _scatter(self, vec, mode, i=-1):
        """ Perform scatter for ith subsystem or a full scatter if i = -1 """
        if vec == 'vVec':
            vec1, vec2 = self.vVarPETSc, self.vArgPETSc
        elif vec == 'xVec':
            vec1, vec2 = self.xVarPETSc, self.xArgPETSc
        else:
            raise Exception('vec type not recognized')

        if mode == 'fwd':
            vec1, vec2 = vec1, vec2
            scatter = self.scattersFwd[i]
        elif mode == 'rev':
            vec1, vec2 = vec2, vec1
            scatter = self.scattersRev[i]
        else:
            raise Exception('mode type not recognized')

        if i == -1:
            if self.scatterFull == None:
                return
            else:
                scatter = self.scatterFull

        scatter.scatter(vec1, vec2, addv = True, mode = mode == 'rev')

    def _createPETScVec(self):
        """ Returns a PETSc Vec with a size of the total number of unknowns """
        m = numpy.sum(self.varSizes[self.rank,:])
        return PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)

    def initialize(self, comm=MPI.COMM_WORLD):
        """ Top-level initialize method called by user """
        self._initializeCommunicators(comm)
        self._initializeSizeArrays()

        m = numpy.sum(self.varSizes[self.rank,:])
        zeros = numpy.zeros
        self._initializePETScVecs(zeros(m), zeros(m), zeros(m), zeros(m))
        self._initializePETScScatters()

        self._initializeVecs()



class SimpleSystem(System):
    """ Nonlinear system with only one variable """

    def __init__(self, copy=0, **kwdargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        name, numReqProcs = self._declare()
        self._initializeSystem(name, copy, numReqProcs, kwdargs)
        self.variables[name, copy] = {'size':0, 'args':OrderedDict()}

    def _setLocalSize(self, size):
        """ Size of the variable vector on the current proc """
        self.variables[self.name, self.copy]['size'] = size

    def _setArgument(self, name, indices, copy=0):
        """ Helper method for declaring arguments """
        copy = self.copy if copy == -1 else copy
        arg = self.variables[self.name, self.copy]['args']
        arg[name,copy] = numpy.array(indices,'i')

    def _initializeCommunicators(self, comm):
        """ No subsystems to distribute comms to """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self._declareArguments()
        self.localSubsystems = []

    def _initializePETScScatters(self):
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
            self.scatterFull = PETSc.Scatter().create(self.vVarPETSc, ISvar,
                                                      self.vArgPETSc, ISarg)
        else:
            self.scatterFull = None



class ImplicitVariable(SimpleSystem):
    """ Variable implicitly defined by v_i : C_i(v) = 0 """
    pass



class ExplicitVariable(SimpleSystem):
    """ Variable explicitly defined by v_i : V_i(v_{j!=i}) """

    def _evalC(self):
        """ C_i(v) = v_i - V_i(v_{j!=i}) = 0 """
        self.cVarPETSc.array[:] = 0.0
        self._solve(self)

    def _applyJ(self, mode, arguments):
        """ y = d{C_i}dv * x = x_i - d{V_i}dv * x """
        """ x = d{C_v}dv^T * y = [0,...,y,...,0] - d{V_i}dv^T * y """
        if mode == 'fwd':
            self.yVarPETSc.array[:] += self.xVarPETSc.array[:]
            self.yVarPETSc.array[:] *= -1.0
            self._apply_Jacobian(arguments)
            self.yVarPETSc.array[:] *= -1.0
        elif mode == 'rev':
            self.xVarPETSc.array[:] += self.yVarPETSc.array[:]
            self.yVarPETSc.array[:] *= -1.0
            self._apply_Jacobian_T(arguments)
            self.yVarPETSc.array[:] *= -1.0
        else:
            raise Exception('mode type not recognized')

    def _evalCinv(self):
        """ v_i = V_i(v_{j!=i}) """
        self._solve(self)



class IndependentVariable(SimpleSystem):
    """ Variable given by v_i = v_i^* """

    def _evalC(self):
        """ C_i(v) = v_i - v_i^* = 0 """
        self.cVarPETSc.array[:] = 0.0
        self.vVarPETSc.array[:] = self.value[:]

    def _applyJ(self, mode, arguments):
        """ y = d{C_i}dv * x = x_i """
        """ x = d{C_i}dv^T * y = [0,...,y,...,0] """
        if mode == 'fwd':
            self.yVarPETSc.array[:] += self.xVarPETSc.array[:]
        elif mode == 'rev':
            self.xVarPETSc.array[:] += self.yVarPETSc.array[:]
        else:
            raise Exception('mode type not recognized')

    def _evalCinv(self):
        """ v_i = v_i^* """
        self.vVarPETSc.array[:] = self.value[:]

    def _applyJinv(self, mode):
        """ x = d{C_i}dv^{-1} * y """
        """ y = d{C_i}dv^{-T} * x """
        if mode == 'fwd':
            self.xVarPETSc.array[:] = self.yVarPETSc.array[:]
        elif mode == 'rev':
            self.yVarPETSc.array[:] = self.xVarPETSc.array[:]
        else:
            raise Exception('mode type not recognized')



class CompoundSystem(System):
    """ Nonlinear system with multiple variables; concatenation of subsystems """

    def __init__(self, name, subsystems, copy=0, **kwdargs):
        """ Defines basic attributes and initializes variables/arguments dicts """
        self.subsystems = subsystems
        numReqProcs = numpy.sum([system.numReqProcs for system in subsystems])
        self._initializeSystem(name, copy, numReqProcs, kwdargs)
        for system in subsystems:
            for n,c in system.variables:
                self.variables[n,c] = system.variables[n,c]

    def _initializePETScScatters(self):
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
            return PETSc.Scatter().create(self.vVarPETSc, ISvar, self.vArgPETSc, ISarg)

        variableIndex = self.variables.keys().index
        self.scattersFwd = []
        self.scattersRev = []
        varIndsFull = []
        argIndsFull = []
        i1, i2 = 0, 0
        for system in self.subsystems:
            varIndsFwd = []
            argIndsFwd = []
            varIndsRev = []
            argIndsRev = []
            for (n1, c1) in system.variables:
                args = system.variables[n1,c1]['args']
                for (n2, c2) in args:
                    if (n2, c2) not in system.variables \
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
            self.scattersFwd.append(createScatter(self, varIndsFwd, argIndsFwd))
            self.scattersRev.append(createScatter(self, varIndsRev, argIndsFwd))
        self.scatterFull = createScatter(self, varIndsFull, argIndsFull)

        for system in self.localSubsystems:
            system._initializePETScScatters()

    def _evalC(self):
        """ Delegate to subsystems """
        for system in self.localSubsystems:
            system._evalC()

    def _applyJ(self, mode, arguments):
        """ Delegate to subsystems """
        if mode == 'fwd':
            for system in self.localSubsystems:
                system._applyJ(arguments)
        elif mode == 'rev':
            for system in self.localSubsystems:
                system._applyJT(arguments)
        else:
            raise Exception('mode type not recognized')

    def _computeLinearResidual(self, rhs, mode):
        if mode == 'fwd':
            vec = self.yVarPETSc
        elif mode == 'rev':
            vec = self.xVarPETSc
        else:
            raise Exception('mode type not recognized')
        self.xArgPETSc.array[:] = 0.0
        vec.array[:] = -rhs[:]
        self._scatter('xVec', mode) if mode == 'fwd' else None
        self.applyJ(mode, self.variables.keys())
        self._scatter('xVec', mode) if mode == 'rev' else None
        return vec.norm()



class ParallelSystem(CompoundSystem):

    def _initializeCommunicators(self, comm):
        """ Splits available procs among subsystems based on # requested procs """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

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

        for system in self.localSubsystems:
            system._initializeCommunicators(subcomm)

    def _evalCinv(self):
        """ Jacobi by default """
        self._evalCinv_Jacobi()

    def _evalCinv_blockDiagSolve(self):
        """ Solve each subsystem in parallel """
        self._scatter('vVec', 'fwd')
        for system in self.localSubsystems:
            system._evalCinv()

    def _evalCinv_Jacobi(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Jacobi """
        counter = 0
        norm0 = self.cVarPETSc.norm()
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._evalCinv_blockDiagSolve()
            self._evalC()
            norm = self.cVarPETSc.norm()
            counter += 1

    def _applyJinv(self, mode):
        """ Jacobi by default """
        self._applyJinv_Jacobi(mode)

    def _applyJinv_blockDiagSolve(self, mode):
        """ Invert each subsystem's block; block diagonal preconditioner """
        for system in self.localSubsystems:
            system._applyJinv(mode)

    def _applyJinv_Jacobi(self, mode, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Linear block Jacobi """
        if mode == 'fwd':
            rhs = numpy.array(self.yVarPETSc.array)
        elif mode == 'rev':
            rhs = numpy.array(self.xVarPETSc.array)
        else:
            raise Exception('Vec type not recognized')

        counter = 0
        norm0 = computeNorm(self, rhs, mode)
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._applyJinv_blockDiagSolve(mode)
            norm = self.computeLinearResidual(self, rhs, mode)
            counter += 1



class SerialSystem(CompoundSystem):

    def _initializeCommunicators(self, comm):
        """ Passes communicator to subsystems """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.localSubsystems = self.subsystems

        for system in self.localSubsystems:
            system._initializeCommunicators(self.comm)

    def _evalCinv(self):
        """ Gauss Seidel by default """
        self._evalCinv_GS()

    def _evalCinv_FwdBlockSolve(self):
        """ Solve each subsystem sequentially """
        self._scatter('vVec', 'fwd')
        i = 0
        for system in self.localSubsystems:
            self._scatter('vVec', 'fwd', i)
            system._evalCinv()
            i += 1

    def _evalCinv_GS(self, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Nonlinear block Gauss Seidel """
        counter = 0
        norm0 = self.cVarPETSc.norm()
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._evalCinv_FwdBlockSolve()
            self._evalC()
            norm = self.cVarPETSc.norm()
            counter += 1

    def _applyJinv(self, mode):
        """ Jacobi by default """
        self._applyJinv_GS(mode)

    def _applyJinv_blockTrglSolve(self, mode):
        """ Block fwd or rev substitution; block triangular preconditioner """
        self.xArgPETSc.array[:] = 0.0
        if mode == 'fwd':
            i = 0
            for system in self.localSubsystems:
                self._scatter('xVec', mode, i)
                system.yVarPETSc.array[:] *= -1.0
                system._applyJ(mode, self.variables.keys())
                system.yVarPETSc.array[:] *= -1.0
                system._applyJinv(mode)
                i += 1
        elif mode == 'rev':
            i = 0
            for system in self.localSubsystems:
                system.xVarPETSc.array[:] *= -1.0
                self._scatter('xVec', mode, i)
                system.xVarPETSc.array[:] *= -1.0
                system._applyJinv(mode)
                system._applyJ(mode, self.variables.keys())
        else:
            raise Exception('Vec type not recognized')

    def _applyJinv_GS(self, mode, ilimit=100, atol=1e-6, rtol=1e-4):
        """ Linear block Gauss Seidel """
        if mode == 'fwd':
            rhs = numpy.array(self.yVarPETSc.array)
        elif mode == 'rev':
            rhs = numpy.array(self.xVarPETSc.array)
        else:
            raise Exception('Vec type not recognized')

        counter = 0
        norm0 = computeNorm(self, rhs, mode)
        norm = norm0
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._applyJinv_blockTrglSolve(mode)
            norm = self.computeLinearResidual(self, rhs, mode)
            counter += 1
