from tensorflow_probability import distributions as tfd

N = 1000
dists = {"A":{},"B":{}}
samples = []
for i in range(N):
    dists["A"][i] = tfd.Poisson(rate = 1e-6)
    dists["B"][i] = tfd.Poisson(rate = 1e-6)
    #dists += [tfd.Normal(loc = 0, scale = 1)]

joint = tfd.JointDistributionNamed(dists)
samples = joint.sample(N)
print("joint.log_prob =", joint.log_prob(samples))
