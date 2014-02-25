from __future__ import division
import numpy
from problem import *


mainComp = CompoundComponent('main', structure='tridiagonal', subComps=[
        SimpleComponent('c1', size=2),
        SimpleComponent('c2', size=2),
        ])

mainComp.setup()
solver = mainComp.finalize()

solver.initialize()
solver._evalC()
print solver.vVarPETSc.array
