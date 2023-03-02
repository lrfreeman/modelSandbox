# Import Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

random.seed(0)
np.random.seed(0)

class Generate_Synth_Data:
    def __init__(self, 
                 end_time_of_session: int, 
                 synthetic_trial_number: int, 
                 number_of_artificial_spikes: int,
                 number_of_neurons: int):
        
        # User defined parameters
        self.end_time_of_session = end_time_of_session
        self.synthetic_trial_num = synthetic_trial_number
        self.num_of_artificial_spikes = number_of_artificial_spikes
        self.numNeurons = number_of_neurons
        
        # Hard coded parameters
        self.bin_size = 0.1 # 100ms
        self.numBins = int(self.end_time_of_session/self.bin_size)
        self.probabilityOFTunedNeurons = 0.9
        
        # Funcs
        self.generate_unit_tunning()
        self.generate_event_times()
        self.create_spikes()
        
    #Generate random times uniformly across the session
    def generate_event_times(self):
        """
        Generate random times uniformly across the session from the start of a trial to the end of a trial
        """
        self.event_bin_idx = np.random.randint(0, self.numBins + 1, size= self.synthetic_trial_num)

    #Generate an array of synthetic spike ID's of zero
    def generate_unit_tunning(self):
        """
        Is neuron tuned to a stimulus?
        """
    
        self.tunnedNeurons = np.random.choice([0, 1], size = self.numNeurons, p=[1-self.probabilityOFTunedNeurons, self.probabilityOFTunedNeurons])
        assert len(self.tunnedNeurons) == self.numNeurons, "Number of neurons is not equal to the number of neurons in the tunning array"
    
    # Generate a binned spike train for each neuron for each trial
    def create_spikes(self):
        """
        Create synthetic spike trains using a gamma distribution of shape number of time bins
        """
        
        spike_matrix = np.zeros((self.numBins, self.numNeurons, self.synthetic_trial_num))
                
        for trial in range(self.synthetic_trial_num):
            for neuron in range(self.numNeurons):
                for bin in range(self.numBins):
                    spike_matrix[bin, neuron, trial] = random.randint(0, 10)
                    
                    if self.tunnedNeurons[neuron] == 1 and (self.event_bin_idx[trial] == bin):
                        spike_matrix[bin, neuron, trial] += random.randint(10, 30)
                    
        self.spike_matrix = spike_matrix
    
    def plot_spike_trains_per_trial(self, num_trials_to_plot: int, num_neurons_to_plot: int):
        
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(nrows = num_trials_to_plot, 
                      ncols = 2, 
                      figure = fig)
        
        axs = [fig.add_subplot(gs[i, 0]) for i in range(num_trials_to_plot)]
        crosscorr_axs = [fig.add_subplot(gs[i, 1]) for i in range(num_trials_to_plot)]
        
        for trial in range(num_trials_to_plot):
            dataframe = pd.DataFrame(self.spike_matrix[:, :num_neurons_to_plot, trial])
            sns.lineplot(data=dataframe, ax=axs[trial], legend=False)
            axs[trial].axvline(x=self.event_bin_idx[trial], color='black', linestyle='-')
            axs[trial].set_title(f'Trial {trial}')
            
            # Plot cross-correlogram
            crosscorr = np.zeros((self.spike_matrix.shape[1], self.spike_matrix.shape[1]))
            for neuron1 in range(self.spike_matrix.shape[1]):
                for neuron2 in range(neuron1+1, self.spike_matrix.shape[1]):
                    crosscorr[neuron1, neuron2] = np.correlate(self.spike_matrix[:, neuron1, trial], self.spike_matrix[:, neuron2, trial], mode='full')[len(self.spike_matrix)-1]
                    crosscorr[neuron2, neuron1] = crosscorr[neuron1, neuron2]
            sns.heatmap(data=crosscorr, ax=crosscorr_axs[trial], cmap='coolwarm', cbar=True)
            crosscorr_axs[trial].set_title(f'Trial {trial} ')
            # crosscorr_axs[trial].set_xlabel('Neuron 1')
            # crosscorr_axs[trial].set_ylabel('Neuron 2')
        
        plt.show()
        


    
synthObj = Generate_Synth_Data(end_time_of_session = 10, 
                               synthetic_trial_number = 100, 
                               number_of_artificial_spikes = 1000,
                               number_of_neurons = 5)


synthObj.plot_spike_trains_per_trial(num_trials_to_plot = 4, num_neurons_to_plot = 6)

# trial = 2
# dataframe = pd.DataFrame(spikes[:, :, trial])
# sns.lineplot(data=dataframe)
# plt.axvline(x=synthObj.event_bin_idx[trial], color='black', linestyle='-')
# plt.tight_layout()
# plt.show()