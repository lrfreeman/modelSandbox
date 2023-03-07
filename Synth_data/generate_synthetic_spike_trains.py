# Author: Laurence Freeman - 2022: Designed for the purposes of testing out analysis methods when the underlying factors influencing the data are known.
# TODO: Add unit tests, more stimuli, convert to times, have some neurons negatively tuned to stim etc.

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

class Generate_Synth_Data:
    """
    A Class that generates synthetic spike trains for a given number of neurons, trials and stimuli. Some neurons are tuned to the stimulus and some are not.
    """
    def __init__(self, 
                 number_of_neurons = 10, 
                 number_of_stimuli = 1, 
                 number_of_tuned_neurons = 2,
                 number_of_trials = 650,
                 length_of_trial = 10, # seconds
                 bin_size = 0.1): # seconds 
        
        # Var init
        self.number_of_neurons = number_of_neurons
        self.number_of_stimuli = number_of_stimuli
        self.number_of_tuned_neurons = number_of_tuned_neurons
        self.number_of_trials = number_of_trials
        self.length_of_trial = length_of_trial
        self.bin_size = bin_size
        self.number_of_bins = int(self.length_of_trial / self.bin_size)
        self.time = np.arange(0, self.length_of_trial, self.bin_size)
        self.rate_parameter = 15 # Event occurance rate per bin normal osscilation
        self.rate_paramter_modifier = 15 # Modify the rate parameter for the stimulus bump
        self.plot_trial_num = 6 # how many trials to plot
        
        # Funcs
        self.generate_event_times()
        self.stimMatrix = self.generate_neuron_stim_weight_matrix()
        self.spikeTrains = self.generate_spike_events_for_all_neurons_and_trials()
        self.plot_spike_events()
        
    def generate_neuron_stim_weight_matrix(self) -> np.ndarray:
        """
        This functions creates a matrix of shape (Neuron #, Stimulus #). Where each N_i, S_j decides 
        the weight of the stimulus connected to the neuron.
        """
        # Init the matrix and the tuning weights
        stimMatrix = np.zeros((self.number_of_neurons, self.number_of_stimuli))
        tuningWeights = [random.uniform(0, 1) for neuron in range(self.number_of_tuned_neurons)]
        
        # Assign the tuning weights to the neurons
        columnIdx = 0
        rand_rows = np.random.choice(stimMatrix.shape[0], size = self.number_of_tuned_neurons, replace=False)
        stimMatrix[rand_rows, columnIdx] = tuningWeights
        
        return stimMatrix
    
    def generate_event_times(self) -> None:
        """
        Generate random stim times uniformly across the session from the start of a trial to the end of a trial
        Returns an array of len(number_of_trials)
        """
        self.event_bin_idx = np.random.randint(0, self.number_of_bins, size = self.number_of_trials)
        
    def generate_spike_events_for_a_trial(self) -> np.ndarray:
        """
        This function generates the spike events for one neuron. The spike events are generated
        using a poisson process. The rate of the poisson process is determined by the tuning weights.
        """
        arr = np.full(shape = self.number_of_bins, fill_value = self.bin_size) # Generate an array of identical bin values for the poisson process
        spikeEvents = np.random.poisson(self.rate_parameter * arr) # Poisson process

        return spikeEvents # How many spikes occured in each bin for one trial one neuron
    
    def generate_spike_events_for_all_neurons_and_trials(self) -> np.ndarray:
        """
        Generate spike events but for each neuron and each trial of shape (Bin #, Neuron #, Trial #)
        """
        spikeTrains = np.zeros((self.number_of_bins, self.number_of_neurons, self.number_of_trials))
                
        for trial in range(self.number_of_trials):
            stimBin = self.event_bin_idx[trial] # Get the bin index of the stimulus
            for neuron in range(self.number_of_neurons):
                
                # If the neuron is not tuned to the stimulus
                if self.stimMatrix[neuron] == 0:
                    spikeTrains[:, neuron, trial] = self.generate_spike_events_for_a_trial()
                
                # # If the neuron is tuned to the stimulus
                elif self.stimMatrix[neuron] != 0:
                    spikeTrains[:, neuron, trial] = self.generate_spike_events_for_a_trial()
                    spikeTrains[:, neuron, trial][stimBin] = spikeTrains[:, neuron, trial][stimBin] \
                    + np.random.poisson(self.rate_parameter * self.rate_paramter_modifier * 0.1) # Add the stimulus bump poisson process to stim bin idx
        
        return spikeTrains
    
    def plot_spike_events(self):
        """
        Plot the spike events for a trial after converting the spike events to a dataframe
        """
        trials_to_plot = 6
        fig, axes = plt.subplots(nrows= trials_to_plot, ncols = 1, figsize = (22, 8))

        for trial in range(trials_to_plot):
            dataframe = pd.DataFrame(self.spikeTrains[:, :, trial])
            dataframe.columns = [f"Neuron {neuron}" for neuron in range(self.number_of_neurons)]
        
            # When was the stimulus presented
            stimIdx = self.event_bin_idx[trial]
        
            # apply the Gaussian filter to each neuron 
            for col in dataframe.columns:
                sigma = random.SystemRandom().randint(1, 2) # set the standard deviation of the Gaussian filter randomly for each neuron
                dataframe[col] = gaussian_filter1d(dataframe[col], sigma=sigma)
        
            sns.lineplot(data = dataframe, ax=axes[trial], legend=False)
            axes[trial].axvline(x=stimIdx, color='red')
            axes[trial].set_title(f'Trial Plot: {trial}')
            axes[trial].set_xlim(0, self.number_of_bins)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    # Prevent the random number generator from changing
    random.seed(0)
    np.random.seed(0)
    
    # Gen the data
    synthObj = Generate_Synth_Data()