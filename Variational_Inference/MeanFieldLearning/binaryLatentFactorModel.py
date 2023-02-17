"""Coding up a binary latent factor model using variational inference
The expecation of a binary latent variable s_i which is either 0 or 1
is given by a multipication of the latent variable and the observed. Such that
if S_i = 0 and lambda_i = 0.5 then the expected value of the observed is 0. Because anything x 0 is 0. 
And if S_i = 1 and lambda_i = 0.5 then the expected value of the observed is 0.5. Because anything x 1 is the same value.
Thus the expectation of state S_i is just equal to lambda_i.

The expectated value of a random process is the sum of the probability of each state multiplied by the value of that state.
Such that EV = (S_i = 0 * lanbda_i) + (S_i = 1 * lambda_i) = lambda_i

"""

# Custom Imports
from genimages import gen_images

# OS Libs
import math
import numpy as np

# Helper functions

def safe_log(argument, tol=1e-100): 
    '''
    Useful helper function to prevent nans. If the log of argument is less 
    than a default tolerance level then assign a value of -1e20. This is to 
    prevent underflow, log(zero) etc.
    
    Returns: + -1e20, OR
             + np.log(argument)
    '''
    return np.where(argument > tol, np.log(argument), -1e20)

class MeanFieldModel:
    def __init__(self, data, epsilon, maxSteps, sigma):
        """A binary latent factor model using variational inference. The model
        has a factored Gaussian oberver and a Bernoulli latent model. There are K
        latents variables and N observations. The model is defined as follows:
        """
        self.data = data # N x D
        self.lambdas = None # N x K - Need to add up to 1
        self.mus = None # D x K
        self.pis = None # 1 x K
        self.epsilon = epsilon
        self.maxSteps = maxSteps
        self.D = self.data.shape[1] # How many dimensions is the data, required for the guassian formula
        self.sigma = sigma # The standard deviation of the guassian distribution
        self.K = 16 # How many latent variables are there
        self.N = self.data.shape[0] # How many observations are there
    
    # Initialise the parameters    
    def initialise_params(self):
        
        """Mattias mentioned initalisiation can result in nans, consider init to identiy matrix"""
        
        # Set a seed for reproducibility
        np.random.seed(0)
        
        #Create an array of K random numbers between 0 and 1
        self.pis = np.random.rand(1, self.K)
        
        #Create an array of N X K random numbers between 0 and 1
        self.lambdas = np.random.rand(self.N, self.K)
        
        #Create an array of D X K random numbers between 0 and 1
        self.mus = np.random.rand(self.N, self.K)
    
    # Define the logistic function
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Compute the first term for the lambda
    def lambda_first_term(self, pi, index):
        """Potentially the log odds ratio, but I'm not sure. Produces a float.

        Args:
            pi (_type_): _description_
            index (_type_): _description_

        Returns:
            float: _description_
        """
        return safe_log(pi[:, index] / (1 - pi[:, index]))
        
    # compute the second term for the lambda
    def lambda_second_term(self, index):
        """For each index that isn't the one of interest,
        summate the multiplication of those vectors and
        return the second term for computing lambda
        
        Notes:
        - Expanding the mus in the sum equation does not change the result

        Args:
        
            index (_type_): _description_
            lamb (_type_): _description_
            mu (_type_): _description_

        Returns:
            Array: of length (N, number of observations)
        """
 
        # Loop through latent variables
        sum = np.sum([ self.lambdas[latent] * self.mus[latent] for latent in range(self.K) if latent != index ])
                
        return np.dot((1 / (self.sigma ** 2)) * (self.data - sum), self.mus[index])
         
    # Compute the third term for the lambda
    def lambda_third_term(self, index):
        """Given that there is a transpose I'm assuimg a dot product.
        If I do (K, ).T (K,) then I get a scalar,
        If I do (expanded MU).T (expanded MU) then I get a (K, K) matrix
        Not sure which is right. 
        
        Result: A float
        
        """
        
        return 1 / (2 * self.sigma**2) * self.mus[index].T @ self.mus[index] 
    
    # Compute the lambdas for the E step
    def compute_lambdas(self, data):
        """The output of the terms into the logistic function is a vector of length N (Number of observations).
        So for each latent variable, produce a vector of length N. And do this for each latent variable.

        Args:
            data (_type_): _description_
        """
        for index in range(self.K):
            self.lambdas[:, index] = self.logistic(                            \
                                     self.lambda_first_term(self.pis, index) + \
                                     self.lambda_second_term(index) -          \
                                     self.lambda_third_term(index)
                                     )
    
    # Compute the free energy
    def compute_free_energy(self):
        
        mu_corr = self.mus.T@self.mus # K x K
        
        # Compute the first term -> Scalar
        sumLambdaLogQuotient = np.sum([self.lambdas[:, index] * safe_log(self.pis[:, index] / self.lambdas[:, index]) for index in range(self.K)])
        
        # Compute the second term -> Scalar
        secondTerm = np.sum((1 - self.lambdas) * safe_log((1 - self.pis) / (1 - self.lambdas)))
        
        # Compute the third term -> Scalar
        thirdTerm = np.sum((self.D / 2) * safe_log(2 * self.pis))
        
        # Compute the fourth term -> Scalar
        fourthTerm = self.D * self.sigma
        
        # Compute the fifth term - maybe error beucase of how I handled sufficient statistics
        fithTerm = np.sum(self.data.T @ self.data) + np.sum(mu_corr) * np.sum(self.lambdas * self.lambdas) 
        
        # Compute the sixth term -> Scalar
        sixthTerm = 2 * np.sum(self.mus.T) * np.sum(self.lambdas * self.data)
        
        # Compute the free energy
        freeEnergy = sumLambdaLogQuotient + secondTerm - thirdTerm - fourthTerm - ((1 / (2 * self.sigma ** 2)) * (fithTerm + sixthTerm))

        return freeEnergy
            
    # Compute the sufficient statistics for the M step
    def compute_sufficient_statistics(self):
        
        #Compute ES
        self.ES = BLFM.lambdas
        
        #COmpute ESS -> compute the outer product of each row of the ES array with itself for each data point 
        self.ESS = np.array([np.outer(row, row) for row in self.ES])
        
        # Regularization constant
        alpha = 1e-3

        # Add regularization constant to diagonal of ESS
        self.ESS += alpha * np.eye(self.ESS.shape[1])
         
        #Assertions
        assert self.ES.shape  == (self.N, self.K), "The shape of ES is not (N, K)"
        assert self.ESS.shape == (self.N, self.K, self.K), "The shape of ESS is not (N, K, K)"

        # Check that ESS is invertible
        determinant = np.linalg.det(self.ESS)
        # assert determinant != 0, "The determinant of ESS is 0. It is not invertible"
    
    # Conduct the M step
    def M_step(self, X, ES, ESS):
        """
        mu, sigma, pie = MStep(X,ES,ESS)

        Inputs:
        -----------------
            X: shape (N, D) data matrix
            ES: shape (N, K) E_q[s]
            ESS: shape (K, K) sum over data points of E_q[ss'] (N, K, K)
                            if E_q[ss'] is provided, the sum over N is done for you.

        Outputs:
        --------
            mu: shape (D, K) matrix of means in p(y|{s_i},mu,sigma)
        sigma: shape (,)    standard deviation in same
            pie: shape (1, K) vector of parameters specifying generative distribution for s
        """
        N, D = X.shape
        if ES.shape[0] != N:
            raise TypeError('ES must have the same number of rows as X')
        # K = ES.shape[1]
        if ESS.shape == (N, self.K, self.K):
            ESS = np.sum(ESS, axis=0)
        if ESS.shape != (self.K, self.K):
            raise TypeError('ESS must be square and have the same number of columns as ES')
        
    
        # Calculate mus
        self.mus = np.dot(np.dot(np.linalg.inv(ESS), ES.T), X).T
        
        # Check for mus
        if self.mus.shape[0] != self.D:
            raise TypeError("The number of rows in mu must equal the number of data points")
        
        if self.mus.shape[1] != self.K:
            raise TypeError("The number of columns in mu must equal the number of latent variables")
        
        #Calculate sigmas
        self.sigma = np.sqrt((np.trace(np.dot(X.T, X)) + np.trace(np.dot(np.dot(self.mus.T, self.mus), ESS))
                        - 2 * np.trace(np.dot(np.dot(ES.T, X), self.mus))) / (N * D))
        
        # Calculate pis
        self.pis = np.mean(ES, axis=0, keepdims=True)
        
        print("M Step complete")
        
        return 
               
if __name__ == "__main__":
    
    # Generate some data
    data = gen_images()
    
    # Create the model
    BLFM = MeanFieldModel(data = data, 
                          epsilon = 0.0001, 
                          maxSteps = 10000, 
                          sigma = 1)
    
    # Initialise the parameters
    BLFM.initialise_params()
    
    # Train the model
    for step in range(BLFM.maxSteps):
        print("\n")
        print("Commencing training step:", step)
        
        BLFM.compute_lambdas(BLFM.data)
        fe = BLFM.compute_free_energy()
        
        print("Free energy:", fe)
        BLFM.compute_sufficient_statistics()
        BLFM.M_step(X = BLFM.data, 
                    ES = BLFM.ES, 
                    ESS = BLFM.ESS) # changed the transpose here to get the right shape could be an error
