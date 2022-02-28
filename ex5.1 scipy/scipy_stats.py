# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:59:18 2022

@author: zhao
"""
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

#Poisson distributiom

mu = 5
x = np.arange(poisson.ppf(0.01, mu),
              poisson.ppf(0.99, mu))
y_pmf = poisson.pmf(x, mu)
y_cdf = poisson.cdf(x, mu)
r = poisson.rvs(mu, size=1000)
ax1 = plt.subplot (311)
ax1.plot(x,y_pmf, 'bo', ms=8)
ax1.set_title("poisson distribution,pmf,cdf,Distribution ")
ax2 = plt.subplot(312)
ax2.plot(x,y_cdf)
ax3 = plt.subplot(313)
plt.hist(r)



