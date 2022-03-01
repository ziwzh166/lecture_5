# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:59:18 2022

@author: zhao
"""
import numpy as np
from scipy.stats import poisson,norm,ttest_ind
import matplotlib.pyplot as plt

#Poisson distributiom

mu1 = 5
x1 = np.arange(poisson.ppf(0.01, mu1),
              poisson.ppf(0.99, mu1))
y_pmf = poisson.pmf(x1, mu1)
y_cdf = poisson.cdf(x1, mu1)
r1 = poisson.rvs(mu1, size=1000)
fig1, (ax1,ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1 = plt.subplot (311)
ax1.plot(x1,y_pmf, 'bo', ms=8)
ax1.set_title("poisson distribution,pmf,cdf,Distribution ")
ax2 = plt.subplot(312)
ax2.plot(x1,y_cdf)
ax3 = plt.subplot(313)
plt.hist(r1)

# normal distributiom

mu2 = 50
x2 = np.arange(norm.ppf(0.01, mu2),
              norm.ppf(0.99, mu2))
y_pmf = norm.pdf(x2, mu2)
y_cdf = norm.cdf(x2, mu2)
r2 = norm.rvs(size=1000,scale = max(x2))
fig2, (ax1,ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1 = plt.subplot (311)
ax1.plot(x2,y_pmf, 'bo', ms=8)
ax1.set_title("norm distribution,pmf,cdf,Distribution ")
ax2 = plt.subplot(312)
ax2.plot(x2,y_cdf)
ax3 = plt.subplot(313)
plt.hist(r2)

# check the sourse distribution of r1 r2
# pvalue different
print(ttest_ind(r1,r2))
print(ttest_ind(r1,r2,equal_var=False))

