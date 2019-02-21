import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 20
b = 200
X = np.random.randn(n)

print("Sample mean of x: ", X.mean())

invidual_estimates = np.empty(b)
for i in range(b):
    sample = np.random.choice(X, size=n)
    invidual_estimates[i] = sample.mean()

bmean = invidual_estimates.mean()
bstd = invidual_estimates.std()

lower = bmean + norm.ppf(0.025) * bstd
upper = bmean + norm.ppf(0.975) * bstd

print("Bootstrping mean of x ", bmean)

lower2 = X.mean() + norm.ppf(0.025)*X.std()/(np.sqrt(n))
upper2 = X.mean() + norm.ppf(0.975)*X.std()/(np.sqrt(n))

plt.hist(invidual_estimates, bins=20)
plt.axvline(x = lower,linestyle = "--", color='g', label = 'lower bound for  95% confidence inteerval')
plt.axvline(x = upper,linestyle = "--", color='g', label = 'lower bound for  95% confidence inteerval')
plt.axvline(x = lower2,linestyle = "--", color='r', label = 'lower bound for  95% confidence inteerval')
plt.axvline(x = upper2,linestyle = "--", color='r', label = 'lower bound for  95% confidence inteerval')
plt.legend()
plt.show()