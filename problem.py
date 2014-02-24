from __future__ import division
import numpy
import random
from system import *
import GCMPlib



class Component(object):
    """ Component base class representing a group of variables """

    def __init__(self, name, copy, **kwargs):
        self.name = name
        self.copy = copy
        self._initialize(kwargs)

        self.superComp = None
        self.iComp = None
        for i in xrange(len(self.subComps)):
            comp = self.subComps[i]
            comp.superComp = self
            comp.iComp = i

        self.varParams = {p:None for p in ['size', 'deg', 'nonlin', 'numArgs']]}
        for param in self.varParams:
            if param in kwargs:
                self.varParams[param] = kwargs[param]

        self.compParams = {p:None for p in ['coupling', 'condNum', 'numDeps', 'struct']}
        for param in self.compParams:
            if param in kwargs:
                self.compParams[param] = kwargs[param]

    def _processCompParams(self):
        defaults = {'coupling':0, 'condNum':1.0, 'numDeps':1, 'struct':'diagonal'}
        for param in self.compParams:
            if self.compParams[param] is None:
                self.compParams[param] = defaults[param]

        params = self.compParams
        if params['struct'] is 'diagonal':
            params['struct'] = [False, 0, 0]
        elif params['struct'] is 'tridiagonal':
            params['struct'] = [False, -1, 1]
        elif params['struct'] is 'upper triangular':
            params['struct'] = [False, 0, len(self.children)-1]
        elif params['struct'] is 'lower triangular':
            params['struct'] = [False, -len(self.children)-1, 0]
        elif params['struct'] is 'upper Hessenberg':
            params['struct'] = [False, -1, len(self.children)-1]
        elif params['struct'] is 'lower Hessenberg':
            params['struct'] = [False, -len(self.children)-1, 1]

    def setup(self):
        self._setup1_VarParams()
        self._setup2_CompParams()
        self._setup3_AdjGraph()
        self._setup4_Variables()
        self._setup5_Arguments()
        self._setup6_Scaling()

    def finalize(self):
        return self._finalize()



class SimpleComponent(Component):

    def _initialize(self, kwargs):
        self.subComps = []

    def _setup1_VarParams(self):
        defaults = {'size':1, 'deg':2, 'nonlin':0.0, 'numArgs':3}
        for param in self.varParams:
            if self.varParams[param] is None:
                self.varParams[param] = defaults[param]

    def _setup2_CompParams(self):
        self._processCompParams()

    def _setup3_AdjGraph(self):
        pass

    def _setup4_Variables(self):
        self.variables = {(self.name, self.copy): self.varParams['size']}

    def _setup5_Arguments(self, args):
        self.args = {}
        for (n,c) in random.sample(args.keys(), min(len(args), self.varParams['numArgs'])):
            self.args[n,c] = args[n,c]

    def _setup6_Scaling(self, scaling):
        self.scaling = scaling

    def _finalize(self):
        class Var(ImplicitVariable):
            def _declare(self):
                self.comp = self.kwargs['comp']
                return self.comp.name, 1
            def _declareArguments(self):
                self.n = self.comp.varParams['size']
                localSizes = numpy.zeros(self.size, int)
                for i in xrange(self.size):
                    procPctg = 1.0/(self.size-i)
                    remainingSize = self.n - numpy.sum(localSizes)
                    localSizes[i] = int(round(procPctg * remainingSize))
                self.i1 = numpy.sum(localSizes[:self.rank])
                self.i2 = numpy.sum(localSizes[:self.rank+1])
                self._setLocalSize(localSizes[self.rank])

                wrap, a, b = self.comp.compParams['struct']
                coupl = self.comp.compParams['coupl']
                numDeps = self.comp.compParams['numDeps']
                rands = numpy.array([random.random() for i in xrange(numDeps)])
                nLocal = self.i2 - self.i1
                indices, cplFactors = GCMPlib.unknownarg(wrap, a, b, self.i1, self.i2, \
                                                             self.n, nLocal, numDeps \
                                                             coupl, rands)
                self._setArgument(self.name, indices=indices.reshape(nLocal*numDeps, order='F'), self.copy)

                args = self.comp.args
                for (n,c) in args:
                    indices = numpy.linspace(0, args[n,c][0]-1, args[n,c][0])
                    self._setArgument(n, indices=indices, c)

        return Var(self.copy, comp=self)



class CompoundComponent(Component):

    def _initialize(self, kwargs):
        self.subComps = kwargs['subComps']
        self.mode = 'serial' if 'mode' not in kwargs else kwargs['mode']

    def _setup1_VarParams(self):
        for comp in self.subComps:
            for param in self.varParams:
                if comp.varParams[param] is None:
                    comp.varParams[param] = self.varParams[param]
            comp._setup1_VarParams()

    def _setup2_CompParams(self):
        self._processCompParams()
        for comp in self.subComps:
            comp._setup2_CompParams()

    def _setup3_AdjGraph(self):
        numDeps = self.compParams['numDeps']
        wrap, a, b = self.compParams['struct']
        coupl = self.compParams['coupl']
        n = len(self.subComp)

        self.edges = {}
        for i in xrange(n):
            self.edges[i] = {}
            js = random.sample(range(i+a,i) + range(i+1,i+b+1), min(b-a, numDeps))
            for j in js:
                if 0 <= j < n:
                    self.edges[i][j] = numpy.exp(-coupl * abs(i-j))
                elif wrap and j < 0:
                    self.edges[i][j+n] = numpy.exp(-coupl * abs(i-j))
                elif wrap and j >= n:
                    self.edges[i][j-n] = numpy.exp(-coupl * abs(i-j))

        for comp in self.subComps:
            comp._setup3_AdjGraph()

    def _setup4_Variables(self):
        self.variables = {}
        for comp in self.subComps:
            comp._setup4_Variables()
            self.variables.update(comp.variables)

    def _setup5_Arguments(self, selfArgs={}):
        for i in xrange(len(self.subComps)):
            args = {k:[selfArgs[k][0], selfArgs[k][1]] for k in selfArgs}
            for j in self.edges[i]:
                n,c = self.subComps[j].name, self.subComps[j].copy
                args[n,c] = [self.variables[n,c], self.edges[i][j]]
            self.subComps[i]._setup5_Arguments(args)

    def _setup6_Scaling(self, selfScaling=1.0):
        cond = self.compParams['condNum']
        scaling = 10**numpy.linspace(-cond/2.0, cond/2.0, len(self.subComps))
        for i in xrange(len(self.subComps)):
            self.subComps[i]._setup6_Scaling(selfScaling * scaling[i])

    def _finalize(self):
        if self.mode is 'parallel':
            system = ParallelSystem
        elif self.mode is 'serial':
            system = SerialSystem

        subSystems = []
        for comp in self.subComps:
            subSystems.append(comp._finalize())

        return system(self.name, subSystems, self.copy)
