"""ALGORITHM 2: Imhomogeneous Poisson process; Taken from Thinning Algorithms for Simulating Point Processes 
by Yuanda Chen, 2016, https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf"""

import matplotlib.pyplot as plt
import numpy as np

def define_time(total_time, time_step):
    return np.arange(0, total_time, time_step)

def sample_inhomogeneous_pp_thinning_v2(cif_function, T):
    """_summary_

    Args:
        cif_function (_type_): _description_
        T (Int): Samples are taken from the interval [0, T]

    Returns:
        _type_: _description_
    """
    n = 0
    m = 0
    point = [0]
    s = [0]
    time = define_time(T, 0.01)
    lambda_bar = max([cif_function(x) for x in time])

    while s[m] < T:
        
        u = np.random.uniform(low = 0.01, high = 1)
        w = -np.log(u) / lambda_bar
        s.append(s[m] + w)
        D = np.random.uniform(low = 0.01, high = 1)

        if D <= cif_function(s[m + 1]) / lambda_bar:
            point.append(s[m + 1])
            n += 1

        m += 1

    if point[-1] <= T:
        return {"inhomogeneous": point[1:], "homogeneous": s[1:]}
    
    else:
        return {"inhomogeneous": point[1:-1], "homogeneous": s[1:-1]}

# Define the Gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Define the cif_function using the Gaussian function with peak_time, width, and peak_intensity parameters
def cif_function_gaussian(t, peak_time, width, peak_intensity):
    return gaussian(t, mu=peak_time, sigma=width) * peak_intensity

# Simulate an inhomogeneous Poisson process with the Gaussian intensity rate function
def simulate_gaussian_intensity(T, peak_time, width, peak_intensity):
    result = sample_inhomogeneous_pp_thinning_v2(lambda t: cif_function_gaussian(t, peak_time, width, peak_intensity), T)
    inhomogeneous = result["inhomogeneous"]
    return inhomogeneous

# Compute hist
def compute_hist(events, T, bins, bin_duration , num_bins):
    # Calculate the bin edges
    bin_edges = np.arange(0, T, bin_duration)[:num_bins+1]
    
    hist, bin_edges = np.histogram(events, bins=bins, range=(0, T))
    
    return hist, bin_edges

# Parameters for the simulation
T = 10 
peak_time_of_kernel = 3 
width_of_kernel = 2 
peak_intensity_of_kernel = 50
num_bins = 10
bin_duration = T / num_bins

# Simulate the inhomogeneous Poisson process
inhomogeneous_events = simulate_gaussian_intensity(T, peak_time_of_kernel, width_of_kernel, peak_intensity_of_kernel)
hist, bin_edges = compute_hist(inhomogeneous_events, T, num_bins, bin_duration, num_bins)

# Create a 1x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 4), sharex=True)

# Plot the event plot
axes[0].eventplot(inhomogeneous_events, linelengths=0.8, color='black')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Events')
axes[0].set_title('Event plot with Gaussian intensity')
axes[0].grid()
axes[0].set_xlim(0, T)

# Plot the histogram
axes[1].bar(bin_edges[:-1], hist, width = bin_duration, align='edge', edgecolor='black')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Event Count')
axes[1].set_title('Event histogram with Gaussian intensity')
axes[1].grid()

# Display the plots
plt.tight_layout()
plt.show()

