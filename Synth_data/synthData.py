# Import Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

class Generate_Synth_Data:
    def __init__(self, 
                 number_of_neurons = 3, 
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
        
        # Funcs
        self.generate_event_times()
        self.stimMatrix = self.generate_neuron_stim_weight_matrix()
        self.spikeTrains = self.generate_spike_events_for_all_neurons_and_trials()
        self.plot_spike_events(1)
        
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
        self.event_bin_idx = np.random.randint(0, self.number_of_bins + 1, size = self.number_of_trials)
        
    def generate_spike_events_for_a_trial(self) -> np.ndarray:
        """
        This function generates the spike events for one neuron. The spike events are generated
        using a poisson process. The rate of the poisson process is determined by the tuning weights.
        """
        arr = np.full(shape = self.number_of_bins, fill_value = self.bin_size) # Generate an array of identical bin values for the poisson process
        lam = 15 # Event occurance rate
        
        # Poisson process
        spikeEvents = np.random.poisson(lam*arr)
        
        return spikeEvents # How many spikes occured in each bin for one trial one neuron
    
    def generate_spike_events_for_all_neurons_and_trials(self) -> np.ndarray:
        """
        Generate spike events but for each neuron and each trial of shape (Bin #, Neuron #, Trial #)
        """
        spikeTrains = np.zeros((self.number_of_bins, self.number_of_neurons, self.number_of_trials))
        
        for trial in range(self.number_of_trials):
            for neuron in range(self.number_of_neurons):                
                spikeTrains[:, neuron, trial] = self.generate_spike_events_for_a_trial()
        
        return spikeTrains
    
    def plot_spike_events(self, trial):
        """
        Plot the spike events for a trial after converting the spike events to a dataframe
        """
        dataframe = pd.DataFrame(self.spikeTrains[:, :, trial])
        dataframe.columns = [f"Neuron {neuron}" for neuron in range(self.number_of_neurons)]
        
        # apply the Gaussian filter
        sigma = 2  # set the standard deviation of the Gaussian filter
        for col in dataframe.columns:
            dataframe[col] = gaussian_filter1d(dataframe[col], sigma=sigma)
        
        sns.lineplot(data = dataframe)
        plt.show()

if __name__ == "__main__":
    
    # Prevent the random number generator from changing
    random.seed(0)
    
    # Gem the data
    synthObj = Generate_Synth_Data()
    
    


# random.seed(0)
# np.random.seed(0)

# class Generate_Synth_Data:
#     def __init__(self, 
#                  end_time_of_session: int, 
#                  synthetic_trial_number: int, 
#                  number_of_artificial_spikes: int,
#                  number_of_neurons: int):
        
#         # User defined parameters
#         self.end_time_of_session = end_time_of_session
#         self.synthetic_trial_num = synthetic_trial_number
#         self.num_of_artificial_spikes = number_of_artificial_spikes
#         self.numNeurons = number_of_neurons
        
#         # Hard coded parameters
#         self.bin_size = 0.1 # 100ms
#         self.numBins = int(self.end_time_of_session/self.bin_size)
#         self.probabilityOFTunedNeurons = 0.45
        
#         # Funcs
#         self.generate_unit_tunning()
#         self.generate_event_times()
#         self.create_spikes()
        
#     #Generate random times uniformly across the session
#     def generate_event_times(self):
#         """
#         Generate random times uniformly across the session from the start of a trial to the end of a trial
#         """
#         self.event_bin_idx = np.random.randint(0, self.numBins + 1, size= self.synthetic_trial_num)
#         print("bin index of event", self.event_bin_idx)

#     #Generate an array of synthetic spike ID's of zero
#     def generate_unit_tunning(self):
#         """
#         Is neuron tuned to a stimulus?
#         """
    
#         self.tunnedNeurons = np.random.choice([0, 1], size = self.numNeurons, p=[1-self.probabilityOFTunedNeurons, self.probabilityOFTunedNeurons])
#         print(self.tunnedNeurons)
#         assert len(self.tunnedNeurons) == self.numNeurons, "Number of neurons is not equal to the number of neurons in the tunning array"
    
#     # Generate a binned spike train for each neuron for each trial
#     def create_spikes(self):
#         """
#         Create synthetic spike trains using a gamma distribution of shape number of time bins
#         """
        
#         spike_matrix = np.zeros((self.numBins, self.numNeurons, self.synthetic_trial_num))
                
#         for trial in range(self.synthetic_trial_num):
#             for neuron in range(self.numNeurons):
#                 for bin in range(self.numBins):
#                     spike_matrix[bin, neuron, trial] = random.randint(0, 10)
                    
#                     if self.tunnedNeurons[neuron] == 1 and self.event_bin_idx[trial] == bin:
#                         spike_matrix[bin, neuron, trial] += random.randint(10, 30)
                    
#         self.spike_matrix = spike_matrix
    
#     def plot_spike_trains_per_trial(self, num_neurons_to_plot: int):
        
#         fig = plt.figure(constrained_layout=True)
#         gs = GridSpec(nrows = self.synthetic_trial_num, 
#                       ncols = 2, 
#                       figure = fig)
        
#         axs = [fig.add_subplot(gs[i, 0]) for i in range(self.synthetic_trial_num,)]
#         crosscorr_axs = [fig.add_subplot(gs[i, 1]) for i in range(self.synthetic_trial_num,)]
        
#         for trial in range(self.synthetic_trial_num,):
#             dataframe = pd.DataFrame(self.spike_matrix[:, :num_neurons_to_plot, trial])
#             sns.lineplot(data=dataframe, ax=axs[trial], legend=False)
#             axs[trial].axvline(x=self.event_bin_idx[trial], color='black', linestyle='-')
#             axs[trial].set_title(f'Trial {trial}')
            
#             # Plot cross-correlogram
#             crosscorr = np.zeros((self.spike_matrix.shape[1], self.spike_matrix.shape[1]))
#             for neuron1 in range(self.spike_matrix.shape[1]):
#                 for neuron2 in range(neuron1+1, self.spike_matrix.shape[1]):
#                     crosscorr[neuron1, neuron2] = np.correlate(self.spike_matrix[:, neuron1, trial], self.spike_matrix[:, neuron2, trial], mode='full')[len(self.spike_matrix)-1]
#                     crosscorr[neuron2, neuron1] = crosscorr[neuron1, neuron2]
#             sns.heatmap(data=crosscorr, ax=crosscorr_axs[trial], cmap='coolwarm', cbar=True)
#             crosscorr_axs[trial].set_title(f'Trial {trial} ')
#             # crosscorr_axs[trial].set_xlabel('Neuron 1')
#             # crosscorr_axs[trial].set_ylabel('Neuron 2')
        
#         plt.show()


# if __name__ == "__main__":
#     synthObj = Generate_Synth_Data(end_time_of_session = 10, synthetic_trial_number = 4, number_of_artificial_spikes = 1000, number_of_neurons = 4)
#     synthObj.plot_spike_trains_per_trial(num_neurons_to_plot = 6)

