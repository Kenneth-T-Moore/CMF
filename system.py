"""
General parallel computational modeling framework
John Hwang, March 2014
"""
# pylint: disable=E1101
from __future__ import division
from mpi4py import MPI
from petsc4py import PETSc
import numpy
from collections import OrderedDict


# System-level data containers

class Vec(OrderedDict):
    """ A dictionary of views to the data of a PETSc Vec """

    def __init__(self, system, array):
        """ Accepts the containing system and the data """
        super(Vec, self).__init__()
        self._system = system
        self.array = array
        self.petsc = self._initialize()

    def _initialize(self):
        """ Implemented in derived classes, VarVec and ArgVec """
        pass


class VarVec(Vec):
    """ A system's unknown vector (contains variables) """

    def _initialize(self):
        """ Populates the dictionary and creates the PETSc Vec """
        system = self._system

        start, end = 0, 0
        for var in system.variables:
            if system.variables[var] is not None:
                end += system.variables[var]['size']
                self[var] = self.array[start:end]
                start += system.variables[var]['size']
        if self.array.shape[0] != end:
            raise Exception('Incorrect VarVec array size')

        return PETSc.Vec().createWithArray(self.array, comm=system.comm)

    def __call__(self, var):
        """ Determines the copy number and gets the data """
        return self[self._system.get_ID(var)]


class ArgVec(Vec):
    """ A system's parameter vector (contains arguments) """

    def _initialize(self):
        """ Populates the dictionary and creates the PETSc Vec """
        system = self._system

        start, end = 0, 0
        for subsystem in system.subsystems['local']:
            for elemsystem in subsystem.subsystems['elem']:
                my_args = OrderedDict()
                args = elemsystem.arguments
                for arg in args:
                    if arg not in subsystem.variables and \
                            arg in system.variables:
                        end += args[arg].shape[0]
                        my_args[arg] = self.array[start:end]
                        start += args[arg].shape[0]
                self[elemsystem.name, elemsystem.copy] = my_args
        if not system.subsystems['local']:
            my_args = OrderedDict()
            args = system.arguments
            for arg in args:
                if arg in system.variables:
                    end += args[arg].shape[0]
                    my_args[arg] = self.array[start:end]
                    start += args[arg].shape[0]
            self[system.name, system.copy] = my_args
        if self.array.shape[0] != end:
            raise Exception('Incorrect ArgVec array size')

        return PETSc.Vec().createWithArray(self.array, comm=system.comm)

    def __call__(self, inp1, inp2=None):
        """ Determines the copy number and gets the data """
        system = self._system

        if not system.subsystems['global']:
            inp0 = [system.name, system.copy]
            return self[system.get_ID(inp0)][system.get_ID(inp1)]
        else:
            return self[system.get_ID(inp1)][system.get_ID(inp2)]


# System classes

class System(object):
    """ Nonlinear system base class """

    def __init__(self, name, copy=0, **kwargs):
        """ Called by __init__ of derived classes """
        self.name = name
        self.copy = copy
        self.kwargs = kwargs

        self.subsystems = {'global': [],
                           'local': [],
                           'elem': [],
                           }
        if 'subsystems' in kwargs:
            self.subsystems['global'] = kwargs['subsystems']

        if 'req_nprocs' not in self.kwargs:
            self.kwargs['req_nprocs'] = 1
            for subsystem in self.subsystems['global']:
                self.kwargs['req_nprocs'] += subsystem.kwargs['req_nprocs']

        methods = {'NL': 'NEWTON',
                   'LN': 'KSP_PC',
                   'PC': 'None',
                   'LS': 'BK_TKG',
                   }
        for problem in methods:
            if problem not in kwargs:
                kwargs[problem] = methods[problem]

        tolerances = {'NL_ilimit': 10, 'NL_atol': 1e-15, 'NL_rtol': 1e-6,
                      'LN_ilimit': 10, 'LN_atol': 1e-10, 'LN_rtol': 1e-4,
                      'PC_ilimit': 10, 'PC_atol': 1e-10, 'PC_rtol': 5e-1,
                      'LS_ilimit': 10, 'LS_atol': 1e-10, 'LS_rtol': 9e-1,
                      }
        for option in tolerances:
            if option not in kwargs:
                kwargs[option] = tolerances[option]

        self.variables = OrderedDict()
        self.arguments = OrderedDict()

        self.comm = None
        self.depth = 0
        self.var_sizes = None
        self.arg_sizes = None

        self.vec = {'p': None, 'u': None, 'f': None,
                    'dp': None, 'du': None, 'df': None, 
                    'dg': None}

        self.app_ordering = None
        self.scatter_full = None
        self.scatter_partial = None

        self.mode = 'fwd'
        self.output = True
        self.sol_vec = None
        self.rhs_vec = None

        self.sol_buf = None
        self.rhs_buf = None

        self.solvers = {'NL': None,
                        'LN': None,
                        'LS': None,
                        }

    def _setup_1of7_comms_assign(self):
        """ Implemented in ParallelSystem and SerialSystem """
        pass

    def _setup_2of7_variables_declare(self):
        """ Implemented in ElementarySystem, ParallelSystem & SerialSystem """
        pass

    def _setup_6of7_scatters_declare(self):
        """ Implemented in ElementarySystem and CompoundSystem """
        pass

    def _setup_6of7_scatters_create(self, var_inds, arg_inds):
        """ Concatenates lists of indices and creates a PETSc Scatter """
        merge = lambda x: numpy.concatenate(x) if len(x) > 0 else []
        var_ind_set = PETSc.IS().createGeneral(merge(var_inds), comm=self.comm)
        arg_ind_set = PETSc.IS().createGeneral(merge(arg_inds), comm=self.comm)
        if self.app_ordering is not None:
            var_ind_set = self.app_ordering.app2petsc(var_ind_set)
        return PETSc.Scatter().create(self.vec['u'].petsc, var_ind_set,
                                      self.vec['p'].petsc, arg_ind_set)

    def _setup_6of7_scatters_linspace(self, start, end):
        """ Return a linspace vector of the right int type for PETSc """
        return numpy.array(numpy.linspace(start, end-1, end-start), 'i')

    def setup_1of7_comms(self, depth):
        """ Receives the communicator and distributes to subsystems """
        self._setup_1of7_comms_assign()
        self.depth = depth

        for subsystem in self.subsystems['local']:
            subsystem.setup_1of7_comms(depth + 1)

    def setup_2of7_variables(self):
        """ Determine global variables """
        for subsystem in self.subsystems['local']:
            subsystem.setup_2of7_variables()

        self._setup_2of7_variables_declare()

    def setup_3of7_sizes(self):
        """ Assembles array of variable and argument sizes """
        rank, size = self.comm.rank, self.comm.size

        self.var_sizes = numpy.zeros((size, len(self.variables)), int)
        for var in self.variables:
            if self.variables[var] is not None:
                ivar = self.variables.keys().index(var)
                self.var_sizes[rank, ivar] = self.variables[var]['size']
        self.comm.Allgather(self.var_sizes[rank, :], self.var_sizes)

        self.arg_sizes = numpy.zeros(size, int)
        for subsystem in self.subsystems['local']:
            for elemsystem in subsystem.subsystems['elem']:
                args = elemsystem.arguments
                for arg in args:
                    if arg not in subsystem.variables and \
                            arg in self.variables:
                        self.arg_sizes[rank] += args[arg].shape[0]
        if not self.subsystems['local']:
            args = self.arguments
            for arg in args:
                if arg in self.variables:
                    self.arg_sizes[rank] += args[arg].shape[0]
        self.comm.Allgather(self.arg_sizes[rank], self.arg_sizes)

        for subsystem in self.subsystems['local']:
            subsystem.setup_3of7_sizes()

    def setup_4of7_vecs(self, u_array, f_array, du_array, df_array):
        """ Creates VarVecs and ArgVecs """
        self.vec['u'] = VarVec(self, u_array)
        self.vec['f'] = VarVec(self, f_array)
        self.vec['du'] = VarVec(self, du_array)
        self.vec['df'] = VarVec(self, df_array)
        self.vec['dg'] = self.vec['df']

        start, end = 0, 0
        for subsystem in self.subsystems['local']:
            end += numpy.sum(subsystem.var_sizes[subsystem.comm.rank, :])
            subsystem.setup_4of7_vecs(u_array[start:end], f_array[start:end],
                                      du_array[start:end], df_array[start:end])
            start += numpy.sum(subsystem.var_sizes[subsystem.comm.rank, :])

        arg_size = self.arg_sizes[self.comm.rank]
        self.vec['p'] = ArgVec(self, numpy.zeros(arg_size))
        self.vec['dp'] = ArgVec(self, numpy.zeros(arg_size))

    def setup_5of7_args(self):
        """ Propagates arg pointers down and up the system hierarchy """
        for subsystem in self.subsystems['local']:
            for elemsystem in subsystem.subsystems['elem']:
                elem = elemsystem.name, elemsystem.copy
                for arg in self.vec['p'][elem]:
                    subsystem.vec['p'][elem][arg] = self.vec['p'][elem][arg]
                    subsystem.vec['dp'][elem][arg] = self.vec['dp'][elem][arg]
            subsystem.setup_5of7_args()
            for elemsystem in subsystem.subsystems['elem']:
                elem = elemsystem.name, elemsystem.copy
                for arg in subsystem.vec['p'][elem]:
                    self.vec['p'][elem][arg] = subsystem.vec['p'][elem][arg]
                    self.vec['dp'][elem][arg] = subsystem.vec['dp'][elem][arg]

    def setup_6of7_scatters(self):
        """ Setup PETSc scatters """
        self._setup_6of7_scatters_declare()

        for subsystem in self.subsystems['local']:
            subsystem.setup_6of7_scatters()

    def setup_7of7_solvers(self):
        """ Setup up PETSc KSP object """
        size = numpy.sum(self.var_sizes[self.comm.rank, :])
        zeros = numpy.zeros
        self.sol_buf = PETSc.Vec().createWithArray(zeros(size), comm=self.comm)
        self.rhs_buf = PETSc.Vec().createWithArray(zeros(size), comm=self.comm)

        self.solvers['NL'] = {'NEWTON': Newton(self),
                              'NLN_JC': NonlinearJacobi(self),
                              'NLN_GS': NonlinearGS(self),
                              }  
        self.solvers['LN'] = {'None': Identity(self),
                              'KSP_PC': KSP(self),
                              'LIN_JC': LinearJacobi(self),
                              'LIN_GS': LinearGS(self),
                              }
        self.solvers['LS'] = {'BK_TKG': Backtracking(self),
                              }

        for subsystem in self.subsystems['local']:
            subsystem.setup_7of7_solvers()

    def scatter(self, vec, subsystem=None):
        """ Perform partial or full scatter """
        var = {'nln': 'u', 'lin': 'du'}[vec]
        arg = {'nln': 'p', 'lin': 'dp'}[vec]
        var_petsc = self.vec[var].petsc
        arg_petsc = self.vec[arg].petsc

        if subsystem == None:
            scatter = self.scatter_full
        else:
            scatter = subsystem.scatter_partial

        if not scatter == None:
            if self.mode == 'fwd':
                scatter.scatter(var_petsc, arg_petsc, addv=False, mode=False)
            elif self.mode == 'rev':
                scatter.scatter(arg_petsc, var_petsc, addv=True, mode=True)
            else:
                raise Exception('mode type not recognized')

    def apply_F(self):
        """ Evaluate function, (p,u) |-> f """
        pass

    def apply_dFdpu(self, arguments):
        """ Apply Jacobian, (dp,du) |-> df [fwd] or df |-> (dp,du) [rev] """
        pass

    def solve_F(self):
        """ Solve f for u, p |-> u """
        kwargs = self.kwargs
        self.solvers['NL'][kwargs['NL']](ilimit=kwargs['NL_ilimit'],
                                         atol=kwargs['NL_atol'],
                                         rtol=kwargs['NL_rtol'])

    def solve_dFdu(self):
        """ Solve Jacobian, df |-> du [fwd] or du |-> df [rev] """
        kwargs = self.kwargs
        self.solvers['LN'][kwargs['LN']](ilimit=kwargs['LN_ilimit'],
                                         atol=kwargs['LN_atol'],
                                         rtol=kwargs['LN_rtol'])

    def solve_precon(self):
        """ Apply preconditioner """
        kwargs = self.kwargs
        self.solvers['LN'][kwargs['PC']](ilimit=kwargs['PC_ilimit'],
                                         atol=kwargs['PC_atol'],
                                         rtol=kwargs['PC_rtol'])

    def solve_line_search(self):
        """ Apply line search """
        kwargs = self.kwargs
        self.solvers['LS'][kwargs['LS']](ilimit=kwargs['LS_ilimit'],
                                         atol=kwargs['LS_atol'],
                                         rtol=kwargs['LS_rtol'])

    def get_ID(self, inp):
        """ Return name, copy even when copy not specified """
        if not isinstance(inp, list):
            return inp, 0
        elif len(inp) == 1:
            return inp[0], 0
        elif inp[1] == -1:
            return inp[0], self.copy
        else:
            return inp[0], inp[1]

    def set_mode(self, mode, output=True):
        """ Set to fwd or rev mode """
        self.mode = mode
        self.output = output
        self.sol_vec = self.vec[{'fwd': 'du', 'rev': 'df'}[mode]]
        self.rhs_vec = self.vec[{'fwd': 'df', 'rev': 'du'}[mode]]

        for subsystem in self.subsystems['local']:
            subsystem.set_mode(mode, output)

    def setup(self, comm=MPI.COMM_WORLD):
        """ Top-level setup/initialization method called by user """
        self.comm = comm
        self.setup_1of7_comms(0)
        self.setup_2of7_variables()
        self.setup_3of7_sizes()

        size = numpy.sum(self.var_sizes[self.comm.rank, :])
        zrs = numpy.zeros
        self.setup_4of7_vecs(zrs(size), zrs(size), zrs(size), zrs(size))
        self.setup_5of7_args()
        self.setup_6of7_scatters()
        self.setup_7of7_solvers()
        self.set_mode('fwd')

        for var in self.variables:
            if self.variables[var] is not None:
                self.vec['u'][var][:] = self.variables[var]['val']

        return self

    def compute(self, output=True):
        """ Solves system """
        self.set_mode('fwd', output)
        self.solve_F()
        return self.vec['u']

    def compute_derivatives(self, mode, var, ind=0, output=True):
        """ Solves derivatives of system (direct/adjoint) """
        self.set_mode('fwd', output)
        self.rhs_vec.array[:] = 0.0

        ivar = self.variables.keys().index(self.get_ID(var))
        ind += numpy.sum(self.var_sizes[:, :ivar])
        ind_set = PETSc.IS().createGeneral([ind], comm=self.comm)
        if self.app_ordering is not None:
            ind_set = self.app_ordering.app2petsc(ind_set)
        ind = ind_set.indices[0]
        self.rhs_vec.petsc.setValue(ind, 1.0, addv=False)

        self.solve_dFdu()
        return self.sol_vec

    def check_derivatives(self, mode):
        """ Check derivatives against FD """
        self.set_mode('fwd')

        self.rhs_vec.array[:] = 1.0
        self.apply_dFdpu(self.variables.keys())
        derivs_user = numpy.array(self.rhs_vec.array)

        self.rhs_vec.array[:] = 1.0
        self._apply_dFdpu_FD(self.variables.keys())
        derivs_FD = numpy.array(self.rhs_vec.array)

        return derivs_user, derivs_FD


class ElementarySystem(System):
    """ Nonlinear system with no subsystems """

    def __call__(self, inp):
        """ Return self if requested name and copy match """
        if self.get_ID(inp) == (self.name, self.copy):
            return self
        else:
            return None

    def _setup_2of7_variables_declare(self):
        """ Calls user's method that declares variables and elemsystems """
        self._declare()
        self.subsystems['elem'] = [self]

    def _declare(self):
        """ Must be implemented by the user """
        raise Exception('This method must be implemented')

    def _declare_variable(self, inp, size=1, val=1.0, lower=None, upper=None):
        """ Adds a variable owned by the current ElementarySystem """
        var = self.get_ID(inp)
        self.variables[var] = {'size': size,
                               'val': val,
                               'lower': lower,
                               'upper': upper,
                               }

    def _declare_argument(self, inp, indices=numpy.zeros(1)):
        """ Adds an argument for the current ElementarySystem """
        arg = self.get_ID(inp)
        self.arguments[arg] = numpy.array(indices, 'i')

    def _setup_6of7_scatters_declare(self):
        """ Defines a scatter for args within an ElementarySystem """
        args = self.arguments
        start = numpy.sum(self.arg_sizes[:self.comm.rank])
        end = numpy.sum(self.arg_sizes[:self.comm.rank+1])
        arg_inds = [self._setup_6of7_scatters_linspace(start, end)]
        var_inds = []
        for arg in args:
            if arg in self.variables:
                ivar = self.variables.keys().index(arg)
                var_inds.append(numpy.sum(self.var_sizes[:, :ivar]) + args[arg])
        self.scatter_full = self._setup_6of7_scatters_create(var_inds, arg_inds)

    def _apply_dFdpu_FD(self, arguments):
        """ Finite difference directional derivative implementation """
        if self.mode == 'rev':
            raise Exception('Not implemented error')
        else:
            step = 1e-3
            vec = self.vec
            sys = self.name, self.copy
            self.scatter('lin')
            self.scatter('nln')

            self.apply_F()
            vec['df'].array[:] = -vec['f'].array

            vec['u'].array[:] += step * vec['du'].array
            for arg in self.arguments:
                if arg in arguments:
                    vec['p'][sys][arg][:] += step * vec['dp'][sys][arg][:]
            self.apply_F()
            vec['u'].array[:] -= step * vec['du'].array
            for arg in self.arguments:
                if arg in arguments:
                    vec['p'][sys][arg][:] -= step * vec['dp'][sys][arg][:]

            vec['df'].array[:] += vec['f'].array
            vec['df'].array[:] /= step

    def apply_dFdpu(self, arguments):
        """ Finite difference directional derivative """
        self._apply_dFdpu_FD(arguments)


class ImplicitSystem(ElementarySystem):
    """ All variables are implicitly defined by v_i : C_i(v) = 0 """
    pass


class ExplicitSystem(ElementarySystem):
    """ All variables are explicitly defined by v_i : V_i(v_{j!=i}) """

    def apply_F(self):
        """ F_i(p_i,u_i) = u_i - G_i(p_i) = 0 """
        self.scatter('nln')
        self.apply_G()
        self.vec['f'].array[:] = 0.0

    def apply_dFdpu(self, arguments):
        """ df = du - dGdp * dp or du = df and dp = -dGdp^T * df """
        vec = self.vec
        if self.mode == 'fwd':
            self.scatter('lin')
            self.apply_dGdp(arguments)
            vec['df'].array[:] *= -1.0
            vec['df'].array[:] += vec['du'].array[:]
        elif self.mode == 'rev':
            self.apply_dGdp(arguments)
            vec['dp'].array[:] *= -1.0
            vec['du'].array[:] = vec['df'].array[:]
            self.scatter('lin')

    def solve_F(self):
        """ v_i = V_i(v_{j!=i}) """
        self.scatter('nln')
        self.apply_G()

    def apply_G(self):
        """ Must be implemented by user """
        pass

    def apply_dGdp(self, arguments):
        """ Optionally implemented by user """
        pass


class IndVar(ExplicitSystem):
    """ Variables given by v_i = v_i^* """

    def __init__(self, name, copy=0, value=0, size=1, **kwargs):
        """ Enables one-line definition of independent variables """
        self.value = value
        if isinstance(self.value, numpy.ndarray):
            self.size = self.value.shape[0]
        else:
            self.size = size

        super(IndVar, self).__init__(name, copy, **kwargs)

    def _declare(self):
        """ Declares the variable """
        self._declare_variable([self.name, self.copy], self.size, self.value)

    def apply_G(self):
        """ Set u to value """
        self.vec['u'][self.name, self.copy][:] = self.value

    def apply_dGdp(self, arguments):
        """ Set to zero """
        if self.mode == 'fwd':
            self.vec['dg'][self.name, self.copy][:] = 0.0


class CompoundSystem(System):
    """ Nonlinear system that has subsystems """

    def __call__(self, inp):
        """ Return instance if found else None """
        for subsystem in self.subsystems['global']:
            result = subsystem(inp)
            if result is not None:
                return result
        return None

    def _setup_6of7_scatters_declare(self):
        """ Defines a scatter for args at this system's level """
        var_sizes = self.var_sizes
        arg_sizes = self.arg_sizes
        iproc = self.comm.rank
        linspace = self._setup_6of7_scatters_linspace
        create = self._setup_6of7_scatters_create

        app_indices = []
        for ivar in xrange(len(self.variables)):
            start = numpy.sum(var_sizes[:, :ivar]) + \
                numpy.sum(var_sizes[:iproc, ivar])
            end = start + var_sizes[iproc, ivar]
            app_indices.append(linspace(start, end))
        app_indices = numpy.concatenate(app_indices)

        start = numpy.sum(var_sizes[:iproc, :])
        end = numpy.sum(var_sizes[:iproc+1, :])
        petsc_indices = linspace(start, end)

        app_ind_set = PETSc.IS().createGeneral(app_indices, comm=self.comm)
        petsc_ind_set = PETSc.IS().createGeneral(petsc_indices, comm=self.comm)
        self.app_ordering = PETSc.AO().createBasic(app_ind_set, petsc_ind_set, 
                                                   comm=self.comm)

        var_full = []
        arg_full = []
        start, end = numpy.sum(arg_sizes[:iproc]), numpy.sum(arg_sizes[:iproc])
        for subsystem in self.subsystems['global']:
            var_partial = []
            arg_partial = []
            for elemsystem in subsystem.subsystems['elem']:
                args = elemsystem.arguments
                for arg in args:
                    if arg not in subsystem.variables and \
                            arg in self.variables:
                        ivar = self.variables.keys().index(arg)
                        var_inds = numpy.sum(var_sizes[:, :ivar]) + args[arg]

                        end += args[arg].shape[0]
                        arg_inds = linspace(start, end)
                        start += args[arg].shape[0]

                        var_partial.append(var_inds)
                        arg_partial.append(arg_inds)
                        var_full.append(var_inds)
                        arg_full.append(arg_inds)
            subsystem.scatter_partial = create(var_partial, arg_partial)

        self.scatter_full = create(var_full, arg_full)

    def apply_F(self):
        """ Delegate to subsystems """
        self.scatter('nln')
        for subsystem in self.subsystems['local']:
            subsystem.apply_F()

    def apply_dFdpu(self, arguments):
        """ Delegate to subsystems """
        if self.mode == 'fwd':
            self.scatter('lin')
        for subsystem in self.subsystems['local']:
            subsystem.apply_dFdpu(arguments)
        if self.mode == 'rev':
            self.scatter('lin')


class ParallelSystem(CompoundSystem):
    """ Distributes procs to subsystems in parallel """

    def _setup_1of7_comms_assign(self):
        """ Splits the comm """
        subsystems = self.subsystems['global']
        rank, size = self.comm.rank, self.comm.size
        nsubs = len(subsystems)
        if nsubs > size:
            raise Exception("Not enough procs to split comm")

        num_procs = numpy.ones(nsubs, int)
        pctg_procs = numpy.zeros(nsubs)
        req_pctg = [subsystem.kwargs['req_nprocs'] for subsystem in subsystems]
        req_pctg = numpy.array(req_pctg, float) / numpy.sum(req_pctg)

        for i in xrange(size - nsubs):
            pctg_procs[:] = num_procs / numpy.sum(num_procs)
            num_procs[numpy.argmax(req_pctg - pctg_procs)] += 1

        color = numpy.zeros(size, int)
        start, end = 0, 0
        for i in xrange(nsubs):
            end += num_procs[i]
            color[start:end] = i
            start += num_procs[i]

        subcomm = self.comm.Split(color[rank])
        self.subsystems['local'] = [subsystems[color[rank]]]

        for subsystem in self.subsystems['local']:
            subsystem.comm = subcomm

    def _setup_2of7_variables_declare(self):
        """ Determine variables and elem subsystems from local subsystems """
        subsystem = self.subsystems['local'][0]
        varkeys_list = self.comm.allgather(subsystem.variables.keys())
        for varkeys in varkeys_list:
            for var in varkeys:
                self.variables[var] = None
        for var in subsystem.variables:
            self.variables[var] = subsystem.variables[var]

        self.subsystems['elem'] = subsystem.subsystems['elem']


class SerialSystem(CompoundSystem):
    """ All subsystems given all procs """

    def _setup_1of7_comms_assign(self):
        """ Passes comm to all subsystems """
        self.subsystems['local'] = self.subsystems['global']

        for subsystem in self.subsystems['local']:
            subsystem.comm = self.comm

    def _setup_2of7_variables_declare(self):
        """ Determine variables and elem subsystems from subsystems """
        for subsystem in self.subsystems['global']:
            for var in subsystem.variables:
                self.variables[var] = subsystem.variables[var]

            self.subsystems['elem'].extend(subsystem.subsystems['elem'])


# Classes for solver collections

class Solver(object):
    """ Base solver class that provides a general iterator method """

    METHOD = ''

    def __init__(self, system):
        """ Accepts a pointer to the containing system """
        self._system = system
        self.info = ''
        self.alpha = 0

    def __call__(self, ilimit=10, atol=1e-6, rtol=1e-4):
        """ Runs the iterator; overwritten for some solvers """
        self._iterator(ilimit, atol, rtol)

    def _iterator(self, ilimit, atol, rtol):
        """ Executes an iterative solver """
        norm0, norm = self._initialize()
        counter = 0
        self.print_info(counter, norm/norm0)
        while counter < ilimit and norm > atol and norm/norm0 > rtol:
            self._operation()
            norm = self._norm()
            counter += 1
            self.print_info(counter, norm/norm0)

    def _norm(self):
        """ Computes the norm that must be driven to zero """
        pass

    def _operation(self):
        """ Operation executed in each iteration """
        pass

    def _initialize(self):
        """ Commands run before iteration """
        norm = self._norm()
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def print_info(self, counter, residual):
        """ Print output from an iteration """
        system = self._system
        if system.comm.rank == 0 and system.output:
            print ('%' + str(3*system.depth) + 's' +
                   '[%-5s,%3i] %s %3i | %.8e %s') \
                   % ('', system.name, system.copy, self.METHOD,
                      counter, residual, self.info)


class NonlinearSolver(Solver):
    """ A base class for nonlinear solvers """

    def _norm(self):
        """ Computes the norm of the f Vec """
        system = self._system
        system.apply_F()
        system.vec['f'].petsc.assemble()
        return system.vec['f'].petsc.norm()


class Newton(NonlinearSolver):
    """ Newton's method with line search """

    METHOD = 'NL: NEWTON'

    def _operation(self):
        """ Find a search direction and apply a line search """
        system = self._system
        system.vec['df'].array[:] = -system.vec['f'].array[:]
        system.solve_dFdu()
        system.solve_line_search()


class Backtracking(NonlinearSolver):
    """ Backtracking line search """

    METHOD = '    LS: BK_TKG'

    def _initialize(self):
        """ Enforce bounds """
        system = self._system
        u = system.vec['u'].array
        du = system.vec['du'].array

        self.alpha = 1.0
        for var in system.variables:
            if system.variables[var] is not None:
                lower = system.variables[var]['lower']
                upper = system.variables[var]['upper']
                if lower is not None:
                    lower_const = u + self.alpha*du - lower
                    ind = numpy.argmin(lower_const)
                    if lower_const[ind] < 0:
                        self.alpha = (lower[ind] - u[ind]) / du[ind]
                if upper is not None:
                    upper_const = upper - u - self.alpha*du
                    ind = numpy.argmin(upper_const)
                    if upper_const[ind] < 0:
                        self.alpha = (upper[ind] - u[ind]) / du[ind]
        self.info = self.alpha

        norm0 = self._norm()
        if norm0 == 0.0:
            norm0 = 1.0
        system.vec['u'].array[:] += self.alpha * system.vec['du'].array[:]
        norm = self._norm()
        return norm0, norm

    def _operation(self):
        """ Return to original u, update alpha, and take new step """
        system = self._system
        system.vec['u'].array[:] -= self.alpha * system.vec['du'].array[:]
        self.alpha /= 2.0
        system.vec['u'].array[:] += self.alpha * system.vec['du'].array[:]
        self.info = self.alpha


class NonlinearJacobi(NonlinearSolver):
    """ Nonlinear block Jacobi """

    METHOD = 'NL: NLN_JC'

    def _operation(self):
        """ Solve each subsystem in parallel """
        system = self._system
        system.scatter('nln')
        for subsystem in system.subsystems['local']:
            subsystem.solve_F()


class NonlinearGS(NonlinearSolver):
    """ Nonlinear block Gauss Seidel """

    METHOD = 'NL: NLN_GS'

    def _operation(self):
        """ Solve each subsystem in series """
        system = self._system
        for subsystem in system.subsystems['local']:
            system.scatter('nln', subsystem)
            subsystem.solve_F()


class LinearSolver(Solver):
    """ A base class for linear solvers """

    def _norm(self):
        """ Computes the norm of the linear residual """
        system = self._system
        system.apply_dFdpu(system.variables.keys())
        system.rhs_vec.array[:] *= -1.0
        system.rhs_vec.array[:] += system.rhs_buf.array[:]
        system.rhs_vec.petsc.assemble()
        return system.rhs_vec.petsc.norm()

    def _initialize(self):
        """ Stores the rhs and initial sol vectors """
        system = self._system
        system.rhs_buf.array[:] = system.rhs_vec.array[:]
        system.sol_buf.array[:] = system.sol_vec.array[:]
        norm = self._norm()
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _finalize(self):
        """ Copy the sol vector ; used for KSP solver """
        system = self._system
        system.sol_vec.array[:] = system.sol_buf.array[:]


class Identity(LinearSolver):
    """ Identity mapping; no preconditioning """

    def __call__(self, ilimit=10, atol=1e-6, rtol=1e-4):
        """ Just copy the rhs to the sol vector """
        system = self._system
        system.sol_vec.array[:] = system.rhs_vec.array[:]


class KSP(LinearSolver):
    """ PETSc's KSP solver with preconditioning """

    METHOD = '   LN: KSP_PC'

    class Monitor(object):
        """ Prints output from PETSc's KSP solvers """

        def __init__(self, ksp):
            """ Stores pointer to the ksp solver """
            self._ksp = ksp
            self._norm0 = 1.0

        def __call__(self, ksp, counter, norm):
            """ Store norm if first iteration, and print norm """
            if counter == 0 and norm != 0.0:
                self._norm0 = norm
            self._ksp.print_info(counter, norm/self._norm0)

    def __init__(self, system):
        """ Set up KSP object """
        super(KSP, self).__init__(system)

        size = numpy.sum(system.var_sizes)
        jac_mat = PETSc.Mat().createPython([size, size], comm=system.comm)
        jac_mat.setPythonContext(self)
        jac_mat.setUp()

        self.ksp = PETSc.KSP().create(comm=system.comm)
        self.ksp.setOperators(jac_mat)
        self.ksp.setType('fgmres')
        self.ksp.setGMRESRestart(10)
        self.ksp.setPCSide(PETSc.PC.Side.RIGHT)
        self.ksp.setMonitor(self.Monitor(self))
            
        pc_mat = self.ksp.getPC()
        pc_mat.setType('python')
        pc_mat.setPythonContext(self)

    def __call__(self, ilimit=10, atol=1e-6, rtol=1e-4):
        """ Run KSP solver """
        system = self._system
        self.ksp.setTolerances(max_it=ilimit, atol=atol, rtol=rtol)
        self._initialize()
        self.ksp.solve(system.rhs_buf, system.sol_buf)
        self._finalize()

    def mult(self, mat, sol_vec, rhs_vec):
        """ Applies Jacobian matrix """
        system = self._system
        system.sol_vec.array[:] = sol_vec.array[:]
        system.apply_dFdpu(system.variables.keys())
        rhs_vec.array[:] = system.rhs_vec.array[:]

    def apply(self, mat, sol_vec, rhs_vec):
        """ Applies preconditioner """
        system = self._system
        system.rhs_vec.array[:] = sol_vec.array[:]
        system.solve_precon()
        rhs_vec.array[:] = system.sol_vec.array[:]


class LinearJacobi(LinearSolver):
    """ Linear block Jacobi """

    METHOD = '   LN: LIN_JC'

    def _operation(self):
        """ Parallel block solve of D x = b - (L+U) x """
        system = self._system

        if system.mode == 'fwd':
            system.scatter('lin')
        for subsystem in system.subsystems['local']:
            args = [v for v in system.variables 
                    if v not in subsystem.variables]
            subsystem.apply_dFdpu(args)
        if system.mode == 'rev':
            system.scatter('lin')

        system.rhs_vec.array[:] *= -1.0
        system.rhs_vec.array[:] += system.rhs_buf.array[:]
        for subsystem in system.subsystems['local']:
            subsystem.solve_dFdu()


class LinearGS(LinearSolver):
    """ Linear block Gauss Seidel """

    METHOD = '   LN: LIN_GS'

    def _operation(self):
        """ Serial block solve of D x = b - (L+U) x """
        system = self._system

        if system.mode == 'rev':
            system.subsystems['local'].reverse()

        system.sol_buf.array[:] = system.rhs_buf.array[:]
        for subsystem in system.subsystems['local']:
            system.rhs_vec.array[:] = 0.0

            if system.mode == 'fwd':
                system.scatter('lin', subsystem)
            args = [v for v in system.variables 
                    if v not in subsystem.variables]
            subsystem.apply_dFdpu(args)
            if system.mode == 'rev':
                system.scatter('lin', subsystem)

            system.sol_buf.array[:] -= system.rhs_vec.array[:]

            system.rhs_vec.array[:] = system.sol_buf.array[:]
            subsystem.solve_dFdu()

        if system.mode == 'rev':
            system.subsystems['local'].reverse()
