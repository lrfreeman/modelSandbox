# Import Libraries
import numpy as np
import random

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
        self.probabilityOFTunedNeurons = 0.2
        
        # Funcs
        self.generate_unit_tunning()
        self.generate_event_times()
        
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
        
    def create_spikes_gama(self):
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
                    
        return spike_matrix
    
synthObj = Generate_Synth_Data(end_time_of_session = 10, 
                               synthetic_trial_number = 100, 
                               number_of_artificial_spikes = 1000,
                               number_of_neurons = 25)

spikes = synthObj.create_spikes_gama()

