import jax
import jax.numpy as np
from jax.scipy.linalg import cho_factor
from jax.config import config
config.update("jax_enable_x64", True)

import json
import matplotlib.pyplot as plt
        
class SquaredExponentialKernel:
    
    def __init__(self, attribute_length_scales: np.ndarray, signal_variance: float):
        self.inverse_L_sqd = np.linalg.inv(np.diag(attribute_length_scales**2))
        self.attribute_length_scales = attribute_length_scales
        self.sqd_signal_variance = signal_variance**2
        
    def __call__(self, x_1: np.ndarray, x_2: np.ndarray):
        return self.sqd_signal_variance * np.exp(-((x_1 - x_2).T @ self.inverse_L_sqd @ (x_1 - x_2)))

class Assn2Kernel:
    def __init__(self, sigma_f: float = 10, length_sqd: float = 16):
        self.sigma_f = sigma_f
        self.length_sqd = length_sqd
    
    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        
        return self.sigma_f * \
                    np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) / (2 * self.length_sqd))


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
        
        kernel = SquaredExponentialKernel(all_hyperparams['attribute_length_scales'], all_hyperparams['signal_variance'])
        #kernel = Assn2Kernel(all_hyperparams['signal_variance']**2, sum(all_hyperparams['attribute_length_scales'])**2)
        K_star_star = np.array([[kernel(a, b) for a in X_val] for b in X_val])
        print(K_star_star)
        
        K_star = np.array([[kernel(a, b) for a in X_val] for b in self.X_train])
        
        K = self.training_kernel(all_hyperparams)
        K_inverse = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(K, lower=True), np.eye(self.n))
        
        posterior_mean = K_star @ K_inverse @ self.y 
        predictive_variance = K_star_star - K_star @ K_inverse  @ K_star.T

        return posterior_mean 
        # key = jax.random.PRNGKey(758493)
        # return [float((p_mean + p_var * jax.random.normal(key, shape=(10000,))).mean())
        #                 for p_mean, p_var in zip(posterior_mean, predictive_variance.diagonal())]

    # --- Compute and return the squared-exponential kernel restricted to just training data: incorporate noise variance here
    def training_kernel(self, all_hyperparams: dict) -> np.ndarray:
        
        sqd_noise_variance = all_hyperparams['noise_variance']**2
        kernel = SquaredExponentialKernel(all_hyperparams['attribute_length_scales'], all_hyperparams['signal_variance'])
        #kernel = Assn2Kernel(all_hyperparams['signal_variance']**2, sum(all_hyperparams['attribute_length_scales'])**2)
        return np.array([[kernel(a, b) for a in self.X_train] for b in self.X_train]) + sqd_noise_variance * np.eye(self.n)
    
    # --- Compute and return the log marginal likelihood. This method should be passed in to your jax.grad function call, in order to compute hyperparameter derivatives
    def log_marginal_likelihood(self, all_hyperparams: dict):
        
        training_K = self.training_kernel(all_hyperparams)  
    
        c, low = cho_factor(training_K)
        K_det = 2 * np.sum(np.log(np.diag(c)))
        inverse_K = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(training_K, lower=True), np.eye(self.n))
        
        data_fit = -0.5 * self.y.T @ inverse_K @ self.y
        model_complexity = 0.5 * K_det         
        return data_fit - model_complexity
    

    def _initialize_hyperparams(self) -> dict:
        
        attribute_length_scales = np.array([160 for i in range(self.X_train.shape[1])])
        #attribute_length_scales = np.array([np.std(self.X_train[:,i]) for i in range(self.X_train.shape[1])])

        noise_variance = np.mean(self.y)
        signal_variance = np.mean(self.y)
        return {'attribute_length_scales': attribute_length_scales, 'noise_variance': noise_variance, 'signal_variance': signal_variance}
        
    # --- Maximize the log marginal likelihood with respect to your hyperparameters using gradient ascent with momentum ; lr is learning rate, and gamma is the momentum term
    def gradient_ascent_marginal_likelihood(self, lr=1e-4, gamma=0.9, n_iters=50):
        
        hyperparams = self._initialize_hyperparams()
        change = {key: np.zeros_like(value) for key, value in hyperparams.items()}
        print(self.log_marginal_likelihood(hyperparams))
        for _ in range(n_iters):
            
            grad = jax.grad(self.log_marginal_likelihood)(hyperparams)
            new_change = {key: gamma * value + lr * grad[key] for key, value in change.items()}
            hyperparams = {key: value + new_change[key] for key, value in hyperparams.items()}
            change = new_change
            print(f"Marginal likelihood: {self.log_marginal_likelihood(hyperparams)}\n")
        
        return hyperparams
    


if __name__=='__main__':
    dataset_prefix = 'concrete'

    X_train = np.load(dataset_prefix+'_train_X.npy')
    X_val = np.load(dataset_prefix+'_val_X.npy')
    y_train = np.load(dataset_prefix+'_train_y.npy')
    y_val = np.load(dataset_prefix+'_val_y.npy')
    attribute_names = json.load(open(dataset_prefix+'_attributes.json','r'))

    X_train = X_train[:100]
    y_train = y_train[:100]
    X_val = X_val[:100]
    y_val = y_val[:100]
    
    gp = GP(X_train, y_train)
    hyperparams = gp.gradient_ascent_marginal_likelihood(n_iters=10)
    predictions = gp.make_predictions(X_val, hyperparams)
    print(predictions)
    print(y_val)
    
    # gp = GP(X_train, y_train)
    # hyperparams = gp.gradient_ascent_marginal_likelihood(n_iters=10)
    # predictions = gp.make_predictions(X_val, hyperparams)
    # print(predictions)
    # print(y_val)
   
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
