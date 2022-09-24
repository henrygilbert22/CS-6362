import jax
import jax.numpy as np
from jax.scipy.linalg import cho_factor,cho_solve

from jax.config import config
config.update("jax_enable_x64", True)

import json
import matplotlib.pyplot as plt

class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length_sqd: float = 1):
        self.sigma_f = sigma_f
        self.length_sqd = length_sqd

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                    np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length_sqd)))
        

# Class for a Gaussian Process squared-exponential kernel, with support for model selection
class GP:
    # --- Constructor: inputs (X), targets (y); Note that hyperparameters will change as optimization progresses, hence their absence in the constructor, and the dependence in remaining methods
    def __init__(self, X_train: np.ndarray, y: np.ndarray):
        self.X_train = X_train
        self.y = y
        self.n = self.X_train.shape[0]
    
    # --- Compute, and store, posterior mean: used to make predictions
    def compute_posterior_mean(self, all_hyperparams: dict):
        pass
    

    # --- Given data (X_val), make predictions -> requires computing the squared-exponential kernel on training and validation data. Note: noise variance should not be used here.
    def make_predictions(self, X_val: np.ndarray, all_hyperparams: dict):
        pass
    

    # --- Compute and return the squared-exponential kernel restricted to just training data: incorporate noise variance here
    def training_kernel(self, all_hyperparams: dict) -> np.ndarray:
        
        L = np.diag(all_hyperparams['attributes_length_scale'])
        noise_variance = all_hyperparams['noise_variance']
        signal_variance = all_hyperparams['signal_variance']
        return (signal_variance**2) * np.exp((self.X_train - self.X_train).T @ np.linalg.inv(L) @ (self.X_train - self.X_train)) + np.diag(noise_variance)**2
    

    # --- Compute and return the log marginal likelihood. This method should be passed in to your jax.grad function call, in order to compute hyperparameter derivatives
    def log_marginal_likelihood(self, all_hyperparams: dict):
        pass
    

    # --- Maximize the log marginal likelihood with respect to your hyperparameters using gradient ascent with momentum ; lr is learning rate, and gamma is the momentum term
    def gradient_ascent_marginal_likelihood(self, all_hyperparams: dict, lr=1e-4, gamma=0.9, n_iters=500):
        pass
    


if __name__=='__main__':
    dataset_prefix = 'concrete'

    X_train = np.load(dataset_prefix+'_train_X.npy')
    X_val = np.load(dataset_prefix+'_val_X.npy')
    y_train = np.load(dataset_prefix+'_train_y.npy')
    y_val = np.load(dataset_prefix+'_val_y.npy')
    attribute_names = json.load(open(dataset_prefix+'_attributes.json','r'))

    # plotting code for the data fit + model complexity terms logged during optimization: assumed data_fit and model_complexity are 1D arrays containing logged values
    '''
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times']})
    plt.rcParams['font.size'] = 12

    fig,ax = plt.subplots()
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('LML Decomposition')
    ax.grid(True, linewidth=0.5)

    opt_steps = np.arange(data_fit.shape[0])
    ax.plot(opt_steps, data_fit, color='tab:red', label='Data fit')
    ax.plot(opt_steps, model_complexity, color='tab:green', label='Model complexity')
    ax.plot(opt_steps, data_fit+model_complexity, color='tab:gray', label='Log marginal likelihood')
    ax.legend()

    plt.show()
    '''

    # plotting code for the hyperparameter importance terms: assumed init_hyperparameters and optimized_hyperparameters are, respectively, hyperparameters initialized and optimized
    '''
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times']})
    plt.rcParams['font.size'] = 10
    ratios = init_hyperparameters / optimized_hyperparameters
    x = np.arange(X_train.shape[1])
    width = .35

    fig, ax = plt.subplots()
    rects = ax.bar(x, ratios, width)

    ax.set_ylabel('Ratio of initialized-to-optimized')
    ax.set_title('Length scale importance')
    ax.set_xticks(x, attribute_names, rotation=90)
    ax.legend()

    fig.tight_layout()

    plt.show()
    '''

#
