import pymc as pm
import numpy as np
from matplotlib import pyplot as plt

A = pm.Uniform("A", lower = 0, upper = 1)
B = pm.Uniform("B", lower = 0 , upper = 1)

@pm.deterministic
def C(A = A, B = B):
    return np.abs(A - B)

model = pm.Model([A,B,C])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

C_trace = mcmc.trace('C')[:]

plt.hist(C_trace, bins = 40)
plt.show()




