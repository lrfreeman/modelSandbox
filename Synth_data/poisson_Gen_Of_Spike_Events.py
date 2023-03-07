"""The rate parameter, λ, is the only number we need to define the Poisson distribution. 
However, since it's a product of two parts (events/interval * interval length), there are two ways to change it: 
we can increase or decrease the events/interval, and we can increase or decrease the interval length."""

# TODO; smoothing the spikes events using a gaussian kernel

import numpy as np
import matplotlib.pyplot as plt

# Number of bins
arr = np.full(shape = 100, fill_value = 0.1)

# Set the rate parameter lambda
lam = 15 # Event occurance rate

# generate the Poisson process
s = np.random.poisson(lam*arr)
print(s)

# set the time interval
T = 10

# Generate the time vector
t = np.arange(0, T, 0.1)

# plot the Poisson process
plt.plot(t, s)
plt.xlabel('Time')
plt.ylabel('Number of events')
plt.show()

