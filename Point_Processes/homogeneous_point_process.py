"""ALGORITHM 1: Homogeneous Poisson process; Taken from Thinning Algorithms for Simulating Point Processes 
by Yuanda Chen, 2016, https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf"""

import numpy as np
import matplotlib.pyplot as plt

def homogeneous_poisson_process(lam, T):
    n = 0
    t0 = 0
    times = []

    while True:
        u = np.random.uniform(low = 0.01, high = 1)
        w = -np.log(u) / lam # So that w ~ Exp(λ)
        tn = t0 + w

        if tn > T:
            return times
        
        else:
            n += 1
            t0 = tn
            times.append(tn)
            
if __name__ == "__main__":

    # Example usage:
    lam = 1.0  # Rate λ
    T = 10.0   # Time interval [0, T]
    result = homogeneous_poisson_process(lam, T)

    # plot the result
    plt.eventplot(result)
    plt.xlabel("Time")
    plt.yticks([]) # Remove the y-axis ticks
    plt.title(f"Homogeneous Poisson process (λ = {lam}, T = {T})")
    plt.show()
