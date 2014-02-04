from __future__ import division
from mpi4py import MPI
from petsc4py import PETSc
import numpy
from collections import OrderedDict



class Component(object):            

    def _initializeAttributes(self, name, copy, kwdargs, numReqProcs):
        class OrderedDict0(OrderedDict):
            def __init__(self, copy):
                self.copy = copy
                super(OrderedDict0,self).__init__()
            def _ID(self, inp):
                copy = 0 if len(inp) is 1 else self.copy if inp[1] is -1 else inp[1]
                return inp[0],copy

        class OrderedDict1(OrderedDict0):            
            def __call__(self, var, arg=None):
                if arg is None:
                    return self[self._ID(var)][None,None]
                else:
                    return self[self._ID(var)][self._ID(arg)]

        class OrderedDict2(OrderedDict0):
            def __call__(self, var):
                return self[self._ID(var)]

        self.name = name
        self.copy = copy
        self.kwdargs = kwdargs
        self.numReqProcs = numReqProcs

        self.variables = OrderedDict()
        self.arguments = OrderedDict()

        self.vVec = OrderedDict1(copy)
        self.xVec = OrderedDict1(copy)
        self.cVec = OrderedDict2(copy)
        self.yVec = OrderedDict2(copy)

    def _initializeSizes(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self._initializeSubCompSizes()
        self._initializeArguments()
        self._initializeVarSizes()
        self._initializeArgSizes()

    def _initializeCommunication(self):
        self._initializeValidArguments()
        self._initializePETScVecs()
        self._initializePETScAO()
        self._initializePETScScatter()
        self._initializeSubCompCommunication()

    def _initializeVecs(self):
        self._initializeDomVecs()
        self._initializeDomVecsDown()
        self._initializeSubCompVecs()
        self._initializeDomVecsUp()
        self._initializeCodVecs()

    def _initializePETScVecs(self):
        m = numpy.sum(self.varSizes[self.rank,:])
        self.vVarPETSc = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)
        self.xVarPETSc = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)

        m = self.argSizes[self.rank]
        self.vArgPETSc = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)
        self.xArgPETSc = PETSc.Vec().createWithArray(numpy.zeros(m), comm=self.comm)

    def _initializeDomVecs(self):
        i1, i2 = 0, 0
        for i in xrange(len(self.variables)):
            n,c = self.variables.keys()[i]
            self.vVec[n,c] = OrderedDict()
            self.xVec[n,c] = OrderedDict()
            i2 += self.varSizes[self.rank,i]
            self.vVec[n,c][None,None] = self.vVarPETSc.array[i1:i2]
            self.xVec[n,c][None,None] = self.xVarPETSc.array[i1:i2]
            i1 = i2

        i1, i2 = 0, 0
        for (n1,c1) in self.arguments:
            for (n2,c2) in self.arguments[n1,c1]:
                i2 += self.arguments[n1,c1][n2,c2][0].shape[0]
                self.vVec[n1,c1][n2,c2] = self.vArgPETSc.array[i1:i2]
                self.xVec[n1,c1][n2,c2] = self.xArgPETSc.array[i1:i2]
                i1 = i2

    def _isValidArgument(self, n1, c1, n2, c2):
        return (n1,c1) in self.variables and (n2,c2) in self.variables and \
            self.variables[n1,c1][0] is not self.variables[n2,c2][0]

    def _evaluate_C(self):
        pass

    def _apply_dCdv(self, arguments):
        pass

    def _apply_dCdv_T(self, arguments):
        pass

    def _evaluate_C_inv(self):
        pass

    def _apply_dCdv_inv(self):
        pass

    def _apply_dCdv_inv_T(self):
        pass

    def _getVec(self, vec, mode):
        if vec is 'vVec':
            vec1, vec2 = self.vVarPETSc, self.vArgPETSc
        elif vec is 'xVec':
            vec1, vec2 = self.xVarPETSc, self.xArgPETSc
        else:
            raise Exception('Vec type not recognized')
        if mode is 'fwd':
            return vec1, vec2
        elif mode is 'rev':
            return vec2, vec1
        else:
            raise Exception('mode type not recognized')

    def _scatterFull(self, vec, mode='fwd'):
        vec1, vec = self._getVec(vec, mode)
        if self.scatterFull is not None:
            self.scatterFull.scatter(vec1, vec2, addv = mode, mode = mode)

    def initialize(self, comm=MPI.COMM_WORLD):
        self._initializeSizes(comm)
        self._initializeCommunication()
        self._initializeVecs()



class SimpleComponent(Component):

    def __init__(self, copy=0, **kwdargs):
        name, numReqProcs = self._declare()
        self._initializeAttributes(name, copy, kwdargs, numReqProcs)
        self.variables[name,copy] = [0,0]
        self.arguments[name,copy] = OrderedDict()

    def _addArgument(self, name, indices, copy=0):
        copy = self.copy if copy is -1 else copy
        self.arguments[self.name,self.copy][name,copy] = [numpy.array(indices,'i'), 0]

    def _initializeSubCompSizes(self):
        pass

    def _initializeArguments(self):
        pass

    def _initializeVarSizes(self):
        self.varSizes = numpy.zeros((self.size, 1),int)
        self.varSizes[self.rank,0] = self.localSize
        self.comm.Allgather(self.varSizes[self.rank,0],self.varSizes[:,0])

    def _initializeArgSizes(self):
        self.argSizes = numpy.zeros(self.size, int)
        n,c = self.name, self.copy
        if (n,c) in self.arguments[n,c]:
            self.argSizes[self.rank] = self.arguments[n,c][n,c][0].shape[0]
        self.comm.Allgather(self.argSizes[self.rank], self.argSizes)

        if (n,c) in self.arguments[n,c]:
            self.arguments[n,c][n,c][1] = numpy.sum(self.argSizes[:self.rank])

    def _initializeValidArguments(self):
        n1,c1 = self.name, self.copy
        for (n2,c2) in self.arguments[n1,c1]:
            if not (n1,c1) == (n2,c2):
                del self.arguments[n1,c1][n2,c2]
 
    def _initializePETScAO(self):
        pass
 
    def _initializePETScScatter(self):
        n,c = self.name, self.copy        
        if (n,c) in self.arguments[n,c]:
            varIndices = self.arguments[n,c][n,c][0]
            m1 = self.arguments[n,c][n,c][1]
            m2 = m1 + self.arguments[n,c][n,c][0].shape[0]   
            argIndices = numpy.linspace(m1,m2-1,m2-m1)
            ISvar = PETSc.IS().createGeneral(numpy.array(varIndices,'i'), comm=self.comm)
            ISarg = PETSc.IS().createGeneral(numpy.array(argIndices,'i'), comm=self.comm)
            self.scatterFull = PETSc.Scatter().create(self.vVarPETSc, ISvar, self.vArgPETSc, ISarg)
        else:
            self.scatterFull = None

    def _initializeSubCompCommunication(self):
        pass
        
    def _initializeDomVecsDown(self):
        pass

    def _initializeSubCompVecs(self):
        pass

    def _initializeDomVecsUp(self):
        pass

    def _initializeCodVecs(self):
        n,c = self.name, self.copy
        self.cVec[n,c] = numpy.zeros(self.varSizes[self.rank,0])
        self.yVec[n,c] = numpy.zeros(self.varSizes[self.rank,0])



class ImplicitVariable(SimpleComponent):

    pass



class ExplicitVariable(SimpleComponent):

    def _evaluate_C(self):
        n,c = self.name, self.copy
        self.cVec[n,c][:] = 0.0
        self._solve(self)

    def _apply_dCdv(self, arguments):
        n,c = self.name, self.copy
        self._apply_Jacobian(arguments)
        self.yVec[n,c][:] *= -1.0
        self.yVec[n,c][:] += self.xVec[n,c][None,None][:]

    def _apply_dCdv_T(self, arguments):
        n,c = self.name, self.copy
        self.yVec[n,c][:] *= -1.0
        self._apply_Jacobian_T(arguments)
        self.yVec[n,c][:] *= -1.0
        self.xVec[n,c][None,None][:] += self.yVec[n,c][:]

    def _evaluate_C_inv(self):
        self._solve(self)



class IndependentVariable(SimpleComponent):
    
    def _evaluate_C(self):
        n,c = self.name, self.copy
        self.cVec[n,c][:] = 0.0
        self.vVec[n,c][:] = self.value[:]

    def _apply_dCdv(self, arguments):
        self.yVec[n,c][:] = self.xVec[n,c][None,None][:]

    def _apply_dCdv_T(self, arguments):
        self.xVec[n,c][None,None][:] += self.yVec[n,c][:]

    def _evaluate_C_inv(self):
        n,c = self.name, self.copy
        self.vVec[n,c][:] = self.value[:]

    def _apply_dCdv_inv(self):
        self.yVec[n,c][:] = self.xVec[n,c][None,None][:]

    def _apply_dCdv_inv_T(self):
        self.xVec[n,c][None,None][:] = self.yVec[n,c][:]
        
                


class MultiComponent(Component):

    def __init__(self, name, subComps, copy=0, **kwdargs):
        self.subComps = subComps
        numReqProcs = numpy.sum([subComp.numReqProcs for subComp in subComps])
        self._initializeAttributes(name, copy, kwdargs, numReqProcs)

        for i in xrange(len(subComps)):
            for n,c in subComps[i].variables:
                self.variables[n,c] = [i,0]
                self.arguments[n,c] = OrderedDict()

    def _initializeArguments(self):
        for subComp in self.subComps:
            for n1,c1 in subComp.arguments:
                for n2,c2 in subComp.arguments[n1,c1]:
                    self.arguments[n1,c1][n2,c2] = [subComp.arguments[n1,c1][n2,c2][i] 
                                                    for i in xrange(2)]

    def _initializeArgSizes(self):
        self.argSizes = numpy.zeros(self.size, int)
        for (n1,c1) in self.arguments:
            for (n2,c2) in self.arguments[n1,c1]:
                if self._isValidArgument(n1,c1,n2,c2):
                    self.argSizes[self.rank] += self.arguments[n1,c1][n2,c2][0].shape[0]
        self.comm.Allgather(self.argSizes[self.rank], self.argSizes)
        
        counter = numpy.sum(self.argSizes[:self.rank])
        for (n1,c1) in self.arguments:
            for (n2,c2) in self.arguments[n1,c1]:
                if self._isValidArgument(n1,c1,n2,c2):
                    self.arguments[n1,c1][n2,c2][1] = counter
                    counter += self.arguments[n1,c1][n2,c2][0].shape[0]

    def _initializeValidArguments(self):
        for (n1,c1) in self.arguments:
            for (n2,c2) in self.arguments[n1,c1]:
                if not self._isValidArgument(n1,c1,n2,c2):
                    del self.arguments[n1,c1][n2,c2]
 
    def _initializePETScAO(self):
        getLinspace = lambda m1, m2: numpy.array(numpy.linspace(m1,m2-1,m2-m1),'i')

        appIndices = []
        for j in xrange(len(self.variables)):
            i = self.rank
            m1 = numpy.sum(self.varSizes[:,:j]) + numpy.sum(self.varSizes[:i,j])
            m2 = m1 + self.varSizes[i,j]
            appIndices.append(getLinspace(m1,m2))
        appIndices = numpy.concatenate(appIndices)

        m1 = numpy.sum(self.varSizes[:self.rank,:])
        m2 = m1 + numpy.sum(self.varSizes[self.rank,:])
        petscIndices = getLinspace(m1,m2)

        ISapp = PETSc.IS().createGeneral(appIndices, comm=self.comm)
        ISpetsc = PETSc.IS().createGeneral(petscIndices, comm=self.comm)
        self.AOvarPETSc = PETSc.AO().createBasic(ISapp, ISpetsc, comm=self.comm)
 
    def _initializePETScScatter(self):
        def createScatter(varInds, argInds):
            merge = lambda x: numpy.array(numpy.concatenate(x) if len(x) > 0 else [],'i')
            ISvar = PETSc.IS().createGeneral(merge(varInds), comm=self.comm)
            ISarg = PETSc.IS().createGeneral(merge(argInds), comm=self.comm)
            ISvar = self.AOvarPETSc.app2petsc(ISvar)
            return PETSc.Scatter().create(self.vVarPETSc, ISvar, self.vArgPETSc, ISarg)            

        self.scattersFwd = []
        self.scattersRev = []
        varIndsFull = []
        argIndsFull = []
        for i in xrange(len(self.subComps)):
            varIndsFwd = []
            argIndsFwd = []
            varIndsRev = []
            argIndsRev = []
            for (n1,c1) in self.subComps[i].arguments:
                for (n2,c2) in self.arguments[n1,c1]:
                    i1 = self.arguments[n1,c1][n2,c2][1]
                    i2 = i1 + self.arguments[n1,c1][n2,c2][0].shape[0]   
                    varInds = self.variables[n2,c2][1] + self.arguments[n1,c1][n2,c2][0]
                    argInds = numpy.linspace(i1,i2-1,i2-i1)
                    if self.variables[n1,c1][0] > self.variables[n2,c2][0]:
                        varIndsFwd.append(varInds)
                        argIndsFwd.append(argInds)
                    else:
                        varIndsRev.append(varInds)
                        argIndsRev.append(argInds)
                    varIndsFull.append(varInds)
                    argIndsFull.append(argInds)
            self.scattersFwd.append(createScatter(varIndsFwd, argIndsFwd))
            self.scattersRev.append(createScatter(varIndsRev, argIndsFwd))
        self.scatterFull = createScatter(varIndsFull, argIndsFull)
        
    def _initializeDomVecsDown(self):
        for subComp in self.subComps:
            for n1,c1 in subComp.vVec:
                for n2,c2 in self.vVec[n1,c1]:
                    if n2 is not None and c2 is not None:
                        subComp.vVec[n1,c1][n2,c2] = self.vVec[n1,c1][n2,c2]
        
    def _initializeDomVecsUp(self):
        for subComp in self.subComps:
            for n1,c1 in subComp.vVec:
                for n2,c2 in subComp.vVec[n1,c1]:
                    if n2 is not None and c2 is not None:
                        self.vVec[n1,c1][n2,c2] = subComp.vVec[n1,c1][n2,c2]

    def _initializeCodVecs(self):
        for subComp in self.subComps:
            for n,c in subComp.cVec:
                self.cVec[n,c] = subComp.cVec[n,c]
                self.yVec[n,c] = subComp.yVec[n,c]

    def _scatterFwd(self, i, vec, mode='fwd'):
        vec1, vec = self._getVec(vec, mode)
        self.scattersFwd[i].scatter(vec1, vec2, addv = rev, mode = rev)

    def _scatterRev(self, i, vec, mode='fwd'):
        vec1, vec = self._getVec(vec, mode)
        self.scattersRev[i].scatter(vec1, vec2, addv = rev, mode = rev)



class ParallelComponent(MultiComponent):

    def _initializeSubCompSizes(self):
        if len(self.subComps) > self.size:
            raise Exception("Not enough procs for parallel decomposition:" + self.name)

        nSubComps = len(self.subComps)
        numProcs = numpy.ones(nSubComps, int)
        pctgProcs = numpy.zeros(nSubComps)
        pctgProcsTarget = numpy.array([subComp.numReqProcs 
                                       for subComp in self.subComps],float)/self.numReqProcs
        for i in xrange(self.size - nSubComps):
            pctgProcs[:] = numProcs/numpy.sum(numProcs)
            index = numpy.argmax(pctgProcsTarget - pctgProcs)
            numProcs[index] += 1

        self.color = numpy.zeros(self.size, int)
        for i in xrange(nSubComps):
            i1, i2 = numpy.sum(numProcs[:i]), numpy.sum(numProcs[:i+1])
            self.color[i1:i2] = i

        self.iSubComp = self.color[self.rank]
        self.numProcs = numProcs
        child_comm = self.comm.Split(self.iSubComp)
        self.subComps[self.iSubComp]._initializeSizes(child_comm)

    def _initializeVarSizes(self):
        numVars = numpy.array([len(subComp.variables) for subComp in self.subComps], int)
        iproc1 = numpy.sum(self.numProcs[:self.iSubComp])
        iproc2 = numpy.sum(self.numProcs[:self.iSubComp+1])
        ivar1 = numpy.sum(numVars[:self.iSubComp])
        ivar2 = numpy.sum(numVars[:self.iSubComp+1])

        self.varSizes = numpy.zeros((self.size, len(self.variables)),int)
        self.varSizes[iproc1:iproc2,ivar1:ivar2] = self.subComps[self.iSubComp].varSizes[:,:]
        self.comm.Allgather(self.varSizes[self.rank,:],self.varSizes)

        counter = 0
        for n,c in self.variables:
            self.variables[n,c][1] = numpy.sum(self.varSizes[:,:counter])
            counter += 1

    def _initializeSubCompCommunication(self):
        self.subComps[self.iSubComp]._initializeCommunication()

    def _initializeSubCompVecs(self):
        self.subComps[self.iSubComp]._initializeVecs()

    def _localCopy(self, subComp, vec, mode='up'):
        if mode is 'up':
            compFrom, compTo = subComp, self
        elif mode is 'down':
            compTo, compFrom = subComp, self
        else:
            raise Exception('mode type not recognized')

        if vec is 'vVec':
            vecFrom, vecTo = compFrom.vVec, compTo.vVec
        elif vec is 'xVec':
            vecFrom, vecTo = compFrom.xVec, compTo.xVec
        elif vec is 'cVec':
            vecFrom, vecTo = compFrom.cVec, compTo.cVec
        elif vec is 'yVec':
            vecFrom, vecTo = compFrom.yVec, compTo.yVec
        else:
            raise Exception('Vec type not recognized')

        for n,c in subComp.variables:
            vecTo(n,c)[:] = vecFrom(n,c)[:]

    def _evaluate_C(self):
        self.subComps[self.iSubComp]._evaluate_C()

    def _apply_dCdv(self, arguments):
        self._scatterFull('xVec', 'fwd')
        subComp = self.subComps[self.iSubComp]
        subComp.yVecPETSc.array[:] = 0.0
        subComp._apply_dCdv(arguments)
        self._localCopy(subComp, 'yVec', 'up')

    def _apply_dCdv_T(self, arguments):
        subComp = self.subComps[self.iSubComp]
        subComp.xVecPETSc.array[:] = 0.0
        subComp._apply_dCdv_T(arguments)
        self._scatterFull('xVec', 'rev')
        self._localCopy(subComp, 'yVec', 'up')

    def _nonlinearJacobi(self):
        self._scatterFull('vVec', 'fwd')
        self._localCopy(subComp, 'vVec', 'down')
        subComp = self.subComps[self.iSubComp]
        subComp._evaluate_C_inv()
        self._localCopy(subComp, 'vVec', 'up')
        


class SerialComponent(MultiComponent):

    def _initializeSubCompSizes(self):
        for subComp in self.subComps:
            subComp._initializeSizes(self.comm)

    def _initializeVarSizes(self):
        self.varSizes = numpy.hstack([subComp.varSizes for subComp in self.subComps])

        counter = 0
        for n,c in self.variables:
            self.variables[n,c][1] = numpy.sum(self.varSizes[:,:counter])
            counter += 1

    def _initializeSubCompCommunication(self):
        for subComp in self.subComps:
            subComp._initializeCommunication()

    def _initializeSubCompVecs(self):
        for subComp in self.subComps:
            subComp._initializeVecs()

    def _evaluate_C(self):
        for subComp in self.subComps:
            subComp._evaluate_C()

    def _apply_dCdv(self, arguments):
        self._scatterFull('xVec', 'fwd')
        for subComp in self.subComps:
            subComp.yVecPETSc.array[:] = 0.0
            subComp._apply_dCdv(arguments)
            self._localCopy(subComp, 'yVec', 'up')

    def _apply_dCdv_T(self, arguments):
        for subComp in self.subComps:
            subComp.xVecPETSc.array[:] = 0.0
            subComp._apply_dCdv_T(arguments)
            self._scatterFull('xVec', 'rev')
            self._localCopy(subComp, 'yVec', 'up')

    def _nonlinearJacobi(self):
        for i in xrange(len(self.subComps)):
            subComp = self.subComps[i]
            self._scatterFwd(i, 'vVec', 'fwd')
            self._localCopy(subComp, 'vVec', 'down')
            subComp._evaluate_C_inv()
            self._localCopy(subComp, 'vVec', 'up')
