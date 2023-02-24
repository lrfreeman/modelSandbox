# Need to do cross validation to compute score

import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Generate synthetic spike count data with 10 neurons and 100 time bins
N = 10
T = 100
X = np.random.poisson(5, size=(N, T))

# Perform factor analysis with 3 latent factors
n_components = 3
fa = FactorAnalysis(n_components=n_components, max_iter=500)
fa.fit(X.T) # Transpose X to obtain the expected format of (samples, features)

# Compute average log-likelihood of the training data
print(fa.score(X.T))

# Obtain the factor loadings and transformed data
W = fa.components_
Z = fa.transform(X.T)

# Plot the factor loadings
fig, axs = plt.subplots(figsize=(10, 20))
axs.imshow(W.T, interpolation='none')
axs.set_ylabel('Neuron')
axs.set_xlabel('Factor')
plt.show()
