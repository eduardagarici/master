import pymc as pm

study_hours = pm.Uniform("study_hours", lower = 0, upper = 20)
c = 5

@pm.deterministic
def probability_to_pass(study_hours = study_hours, c = c):
    return study_hours / (study_hours + c)

y = pm.Bernoulli("y", probability_to_pass)

@pm.deterministic
def bayes_decision():
    if probability_to_pass > 1/2:
        return True
    else:
        return False

model = pm.Model([study_hours, c, y, probability_to_pass, bayes_decision])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

study_hours_samples = mcmc.trace('study_hours')[:]
probability_to_pass_samples = mcmc.trace("probability_to_pass")[:]
bayes_decision_samples = mcmc.trace("bayes_decision")[:]
y_samples = mcmc.trace("y")[:]

print("probability to pass:", (y_samples != bayes_decision_samples).mean())
