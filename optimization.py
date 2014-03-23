"""
Framework interface to pyOptSparse
John Hwang, March 2014
"""

from __future__ import division
from pyoptsparse import Optimization as OptProblem
from pyoptsparse import OPT as Optimizer


class Optimization(object):
    """ Automatically sets up and runs an optimization """

    def __init__(self, system):
        """ Takes system containing all DVs and outputs """
        self._system = system
        self._variables = {'dv': {}, 'func': {}}

    def _get_name(self, var_id):
        """ Returns unique string for the variable """
        return var_id[0] + '_' + str(var_id[1])

    def _add_var(self, typ, var, value=0.0, lower=None, upper=None):
        """ Wrapped by next three methods """
        var_id = self._system.get_id(var)
        var_name = self._get_name(var_id)
        self._variables[typ][var_name] = {'ID': var_id,
                                          'value': value,
                                          'lower': lower,
                                          'upper': upper}

    def add_design_variable(self, var, value=0.0, lower=None, upper=None):
        """ Self-explanatory; part of API """
        self._add_var('dv', var, value=value, lower=lower, upper=upper)

    def add_objective(self, var):
        """ Self-explanatory; part of API """
        self._add_var('func', var)

    def add_constraint(self, var, lower=None, upper=None):
        """ Self-explanatory; part of API """
        self._add_var('func', var, lower=lower, upper=upper)

    def obj_func(self, dv_dict):
        """ Objective function passed to pyOptSparse """
        system = self._system
        variables = self._variables

        for dv_name in variables['dv'].keys():
            dv_id = variables['dv'][dv_name]['ID']
            system(dv_id).value = dv_dict[dv_name]

        system.compute(False)

        func_dict = {}
        for func_name in variables['func'].keys():
            func_id = variables['func'][func_name]['ID']
            func_dict[func_name] = system.vec['u'][func_id]

        fail = False

        return func_dict, fail

    def sens_func(self, dv_dict, func_dict):
        """ Derivatives function passed to pyOptSparse """
        system = self._system
        variables = self._variables

        sens_dict = {}
        for func_name in variables['func'].keys():
            func_id = variables['func'][func_name]['ID']

            for ind in xrange(system.vec['u'][func_id].shape[0]):
                system.compute_derivatives('rev', func_id, ind, False)

                sens_dict[func_name] = {}
                for dv_name in variables['dv'].keys():
                    dv_id = variables['dv'][dv_name]['ID']
                    sens_dict[func_name][dv_name] = system.vec['df'][dv_id]

        fail = False

        return sens_dict, fail

    def __call__(self, optimizer, options=None):
        """ Run optimization """
        system = self._system
        variables = self._variables

        opt_prob = OptProblem('Optimization', self.obj_func)
        for dv_name in variables['dv'].keys():
            dv_id = variables['dv'][dv_name]['ID']
            value = variables['dv'][dv_name]['value']
            lower = variables['dv'][dv_name]['lower']
            upper = variables['dv'][dv_name]['upper']
            size = system(dv_id).size
            opt_prob.addVarGroup(dv_name, size, value=value,
                                 lower=lower, upper=upper)
        opt_prob.finalizeDesignVariables()
        for func_name in variables['func'].keys():
            lower = variables['func'][func_name]['lower']
            upper = variables['func'][func_name]['upper']
            if lower is None and upper is None:
                opt_prob.addObj(func_name)
            else:
                opt_prob.addCon(func_name, lower=lower, upper=upper)

        if options is None:
            options = {}

        opt = Optimizer(optimizer, options=options)
        sol = opt(opt_prob, sens=self.sens_func)
        print sol
