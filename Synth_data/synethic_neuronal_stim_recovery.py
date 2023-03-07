import numpy as np
from factor_analyzer import FactorAnalyzer

# Set the number of neurons and the dimensionality of the stimulus
# Angle assumptions 0 to 360 degrees
n_neurons = 50
stim_dim = 1
numStim = 100
n_trials = 650
trial_duration = 8

# Generate random tuning curves for each neuron, of shape (n_neurons, stim_dim)
tuning_curves = np.random.uniform(low = 0, high = 360, size=(n_neurons, stim_dim))
tuning_curves *= 2*np.pi/360.0

# tuningDicNeurons = {i: tuning_curves[i][0] for i in range(len(tuning_curves))}

# Generate random noise for each neuron
noise = np.random.randn(n_neurons, numStim)

# Generate a stimulus with a range of orientations
stimulus = np.linspace(0, 360, numStim).reshape(-1, 1)

# Spike trains
spike_trains = np.zeros((n_neurons, n_trials, len(stimulus)))

# Loop over trials and generate spike trains for each neuron
for trial in range(n_trials):
    for neuron in range(n_neurons):
        for stim in range(len(stimulus)):
            
            # rate = tuning_curves[neuron][0] * stimulus[stim][0] + noise[neuron][stim]
            # spike_trains[neuron, trial, stim] = np.random.poisson(rate * trial_duration)



# Loop over trials and generate spike trains for each neuron
for trial in range(n_trials):
    
    # Generate random noise for each neuron
    noise = np.random.randn(n_neurons, len(stimulus))
    
    # Calculate the firing rate of each neuron in response to the stimulus
    firing_rates = np.outer(tuning_curves, stimulus) + noise
    
    # Generate spike trains for each neuron using a Poisson process
    for neuron in range(n_neurons):
        for stim in range(len(stimulus)):
            rate = firing_rates[neuron, stim]
            spike_trains[neuron, trial, stim] = np.random.poisson(rate * trial_duration)



# # Calculate the firing rate of each neuron in response to the stimulus
# firing_rates = np.dot(tuning_curves, stimulus.T) + noise
# print(firing_rates.shape)


# # Set the number of components to recover
# n_components = 5

# # Create a factor analysis object and fit it to the data
# fa = FactorAnalyzer(n_factors = n_components)
# fa.fit(firing_rates.T)

# # Get the recovered factors (i.e., the tuning curves)
# recovered_tuning_curves = fa.components_.T