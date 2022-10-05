import jax
import jax.numpy as np
from jax.scipy.linalg import cho_factor
from jax.config import config
config.update("jax_enable_x64", True)
from matplotlib import rc

import json
import matplotlib.pyplot as plt
        
class SquaredExponentialKernel:
    
    def __init__(self, attribute_length_scales: np.ndarray, signal_variance: float):
        self.inverse_L_sqd = np.linalg.inv(np.diag(attribute_length_scales**2))
        self.sqd_signal_variance = signal_variance**2
        
    def __call__(self, x_1: np.ndarray, x_2: np.ndarray):
        return self.sqd_signal_variance * np.exp(-((x_1 - x_2).T @ self.inverse_L_sqd @ (x_1 - x_2)))
    
# Class for a Gaussian Process squared-exponential kernel, with support for model selection
class GP:
    # --- Constructor: inputs (X), targets (y); Note that hyperparameters will change as optimization progresses, hence their absence in the constructor, and the dependence in remaining methods
    def __init__(self, X_train: np.ndarray, y: np.ndarray):
        self.X_train = X_train
        self.y = y
        self.n = self.X_train.shape[0]
    
    
    def relu(self, x: np.ndarray):
        return np.maximum(0, x)
    
    def rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return np.sqrt(np.mean((predictions-targets)**2))
    
    # --- Given data (X_val), make predictions -> requires computing the squared-exponential kernel on training and validation data. Note: noise variance should not be used here.
    def make_predictions(self, X_val: np.ndarray, all_hyperparams: dict):
        
        kernel = SquaredExponentialKernel(all_hyperparams['attribute_length_scales'], all_hyperparams['signal_variance'])
        K_star = np.array([[kernel(a, b) for a in self.X_train] for b in X_val])
        
        K = self.testing_kernel(all_hyperparams)
        K_inverse = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(K, lower=True), np.diag(self.y))
 
        posterior_mean = K_star @ K_inverse @ self.y 
        return posterior_mean


    def _create_covariance_matrix(self, X_1: np.ndarray, X_2: np.ndarray, all_hyperparams: dict):
        kernel = SquaredExponentialKernel(all_hyperparams['attribute_length_scales'], all_hyperparams['signal_variance'])
        covariance_matrix = np.array([[kernel(a, b) for a in self.relu(X_1)] for b in  self.relu(X_2)])
        return covariance_matrix
    
    # --- Compute and return the squared-exponential kernel restricted to just training data: incorporate noise variance here
    def training_kernel(self, all_hyperparams: dict) -> np.ndarray:
        sqd_noise_variance = all_hyperparams['noise_variance']**2
        return self._create_covariance_matrix(self.X_train, self.X_train, all_hyperparams) + sqd_noise_variance * np.eye(self.n)
    
    def testing_kernel(self, all_hyperparams: dict) -> np.ndarray:
        return self._create_covariance_matrix(self.X_train, self.X_train, all_hyperparams)
    
    # --- Compute and return the log marginal likelihood. This method should be passed in to your jax.grad function call, in order to compute hyperparameter derivatives
    def log_marginal_likelihood(self, all_hyperparams: dict, data_fits: list=[], model_fits: list=[]) -> float:
        
        training_K = self.training_kernel(all_hyperparams)  
        c, low = cho_factor(training_K)
        K_det = 2 * np.sum(np.log(np.diag(c)))
        inverse_K = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(training_K, lower=True), np.diag(self.y))
        
        data_fit = -0.5 * self.y.T @ inverse_K @ self.y
        model_complexity = - 0.5 * K_det     
        
        data_fits.append(data_fit)
        model_fits.append(model_complexity)
        return - (data_fit + model_complexity)
        
    def _initialize_hyperparams(self) -> dict:
        
        attribute_length_scales = np.array([np.std(self.X_train[:,i]) for i in range(self.X_train.shape[1])])
        noise_variance = np.mean(self.y)
        signal_variance = np.mean(self.y)
        return {'attribute_length_scales': attribute_length_scales, 'noise_variance': noise_variance, 'signal_variance': signal_variance}
        
    # --- Maximize the log marginal likelihood with respect to your hyperparameters using gradient ascent with momentum ; lr is learning rate, and gamma is the momentum term
    def gradient_ascent_marginal_likelihood(self, lr=1e-1, gamma=0.9, n_iters=50):
        
        hyperparams = self._initialize_hyperparams()
        change = {key: np.zeros_like(value) for key, value in hyperparams.items()}
        
        data_fits = []
        model_fits = []

        for _ in range(n_iters):
            
            grad = jax.grad(self.log_marginal_likelihood)(hyperparams)
            new_change = {key: gamma * value + lr * grad[key] for key, value in change.items()}
            hyperparams = {key: value + new_change[key] for key, value in hyperparams.items()}
            change = new_change
            print(f"Marginal likelihood: {self.log_marginal_likelihood(hyperparams, data_fits, model_fits)}\n")
        
        return hyperparams, np.array(data_fits), np.array(model_fits)
    

def load_data(dataset_prefix: str):
    
    X_train = np.load(dataset_prefix+'_train_X.npy')
    X_val = np.load(dataset_prefix+'_val_X.npy')
    y_train = np.load(dataset_prefix+'_train_y.npy')
    y_val = np.load(dataset_prefix+'_val_y.npy')
    attribute_names = json.load(open(dataset_prefix+'_attributes.json','r'))

    X_train = X_train[:5]
    y_train = y_train[:5]
    X_val = X_val[:5]
    y_val = y_val[:5]
    
    return X_train, y_train, X_val, y_val, attribute_names

def plot_lml_decomposition(dataset_prefix: str, data_fit: np.ndarray, model_complexity: np.ndarray):
    
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
    plt.savefig(f'{dataset_prefix}_lml_decomposition.png')

def plot_hyperparameter_importance(
        dataset_prefix: str, init_hyperparameters: dict, optimized_hyperparameters: dict,
        attribute_names: list, X_train: np.ndarray):
   
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
    plt.savefig(f'{dataset_prefix}_hyperparameter_importance.png')
    
def main():
    
    dataset_prefixes = ['concrete', 'skillcraft']
    for dataset_prefix in dataset_prefixes:

        X_train, y_train, X_val, y_val, attribute_names = load_data(dataset_prefix)
  
        gp = GP(X_train, y_train)
        gp.test_make_kernel(gp._initialize_hyperparams(), gp.X_train, gp.X_train)
        return
        hyperparams, data_fit, model_complexity = gp.gradient_ascent_marginal_likelihood()
        predictions = gp.make_predictions(X_val, hyperparams)
        print(f"Validation RMSE for {dataset_prefix}: {gp.rmse(predictions, y_val)}")
        
        plot_lml_decomposition(dataset_prefix, data_fit, model_complexity)
        plot_hyperparameter_importance(
            dataset_prefix,
            gp._initialize_hyperparams()["attribute_length_scales"], 
            hyperparams["attribute_length_scales"], 
            attribute_names, 
            X_train)
    
    
if __name__=='__main__':
    main()
