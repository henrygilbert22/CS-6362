import json
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import logging
from typing import List
import pandas as pd
import os
import multiprocessing as mp

# TODO (henry): Refactor this
class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length_sqd: float = 1):
        self.sigma_f = sigma_f
        self.length_sqd = length_sqd

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        
        return float(self.sigma_f *
                    np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) / (2 * self.length_sqd)))
        
# Class for a Gaussian Process Squared Exponential Kernel
class GP:
    # --- Constructor: in here, you should compute, and store, the covariance matrix for the given data -- X, a (n x d) matrix -- using passed in hyperparameters called hyperparams
    def __init__(self, X_1, X_2, hyperparams: dict):
        self.X_1 = X_1
        self.X_2 = X_2
        self.hyperparams = hyperparams

        self.squared_kernel = SquaredExponentialKernel(self.hyperparams['signal_variance'], self.hyperparams['sqd_length_scale'])
        self.K = self.generate_covariance_matrix(self.X_1, self.X_2, self.squared_kernel)
   
    # --- compute, and return, the covariance function between data points X_1 and data points X_2
    # ----> if X_1 is a (n1 x d) matrix, and X_2 is a (n2 x d) matrix, then this method should return a (n1 x n2) matrix 
    def generate_covariance_matrix(self, X_1: np.ndarray, X_2: np.ndarray, kernel: Callable) -> np.ndarray:
        """ Square Exponential Covariance Kernel """
        return np.array([[kernel(a, b) for a in X_1] for b in X_2])
    

# Class for computing the Laplace Approximation, and performing binary Gaussian Process classification
class GPClassifier:
    # --- Constructor, should take in:
    #   (1) features (X) -> n x d matrix
    #   (2) class labels (y) -> vector of length n, containing 0 or 1
    #   (3) hyperparameters for the Gaussian process
    # ----> Within the constructor, you should create the Gaussian process for the data, and store it as an instance variable - will be used throughout most methods
   
    mode: float

    def __init__(self, X: np.ndarray , y: np.ndarray, hyperparams: dict):
        self.X = X
        self.y = y
        self.hyperparams = hyperparams
        self.n, self.d = self.X.shape
        self.gp = GP(self.X, self.X, self.hyperparams)

    # --- compute the negative log likelihood for latent function f: this is not necessary, but a very useful debugging tool for your optimization method
    def NLL(self, f: np.ndarray) -> float:
        return -np.sum(self.y*np.log(self.sigmoid(f)) + (1. - self.y)*np.log(1. - self.sigmoid(f)))

    # --- compute classification accuracy for GP classifier probabilities (gp_probs), given ground truth labels (gt_labels)
    @classmethod
    def classification_accuracy(cls, gp_probs: np.ndarray, gt_labels: np.ndarray) -> float:
        return (np.round(gp_probs) == gt_labels).sum() / len(gt_labels)

    # --- compute and return vectorized application of the sigmoid (logistic function) to the given latent function values f
    # ----> Assumption: f is an arbitrary numpy array
    def sigmoid(self, f: np.ndarray) -> np.ndarray:
        return 1./(1. + np.exp(-f))

    # --- compute and return the numerically-stable inversion of the Hessian, via application of the Woodbury matrix inversion lemma
    # ----> K is the GP covariance matrix, and dd_sigmoid is a vector of all second-order partial derivatives of the log likelihood
    def woodbury_inversion(self, K: np.ndarray, dd_sigmoid: np.ndarray) -> np.ndarray:
        
        return K - (K @ np.sqrt(dd_sigmoid)) @ (np.linalg.inv((np.identity(self.n) + (np.sqrt(dd_sigmoid) @ K \
            @ np.sqrt(dd_sigmoid)))) @ (np.sqrt(dd_sigmoid) @ K))

    # --- Newton's method for computing the MAP
    # ----> recommended to store the mode as an instance variable, for later use
    def newton_MAP(self, n_steps=30) -> float:
        # --- recommended initialization for the mode
        
        self.mode = 1e-1*np.random.rand(self.n)
        for _ in range(n_steps):
            
            Hessian_l = self.sigmoid(self.mode)*(1-self.sigmoid(self.mode))
            # Make diagonal matrix from Hessian_l
            z = (Hessian_l * self.mode) + self.y - self.sigmoid(self.mode)
            woodbury_inversion = self.woodbury_inversion(self.gp.K, np.diag(Hessian_l))
            self.mode = woodbury_inversion @ z
        
        logging.info(f"Newton's method: {self.NLL(self.mode)} - {self.hyperparams}")
        return self.NLL(self.mode)
            
    # --- compute the predictive distribution for the latent function - for each point in X_star, this should be the predictive mean, and predictive variance
    def latent_predictive_distribution(self, X_star):
        
        # Should be train and test points
       
        GP_star = GP(X_star, self.X, self.hyperparams)
        GP_star_start = GP(X_star, X_star, self.hyperparams)
        K_star_star = GP_star_start.K
        K_star = GP_star.K
        K = self.gp.K
        
        Hessian_l = self.sigmoid(self.mode)*(1-self.sigmoid(self.mode))
        predictive_variance = K_star_star - K_star.transpose() @ self.woodbury_inversion(np.diag(Hessian_l), K) @ K_star
        predictive_mean = K_star.transpose() @ (self.y - self.sigmoid(self.mode))
        return predictive_mean, predictive_variance.diagonal()

    # --- compute the averaged predictive probability, given a set of predictive distributions (a mean and variance for each example)
    # ----> this should be done via Monte Carlo integration, with `k` being the number of samples (feel free to adjust)
    def averaged_predictive_probability(self, predictive_mean, predictive_variance, k=5000) -> np.array:
        
        return np.array([
            np.mean(self.sigmoid(np.random.normal(p_mean, p_var, k)))
            for p_mean, p_var in zip(predictive_mean, predictive_variance)])


def plot_reject_curve(prefix: str, thresholds: List[float], accuracies: List[float], num_retained: List[int], hyperparams: dict):
   
    # rc('font',**{'family':'serif','serif':['Times']})
    # rc('text', usetex=True)
    # plt.rcParams['font.size'] = 12

    fig,ax = plt.subplots()
    plt_label = f"Squared Length Scale: {hyperparams['sqd_length_scale']}"
    plt.ylim(0,1.01)
    ax.plot(thresholds, accuracies, color='tab:red')
    ax.set_xlabel('Reject Threshold')
    ax.set_ylabel('Accuracy',color='tab:red')

    ax2 = ax.twinx()
    ax2.plot(thresholds, num_retained, color='tab:green')
    ax2.set_ylabel('Number of Samples Retained',color='tab:green')

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#cccccc', linestyle='dashed')
    ax.xaxis.grid(color='#cccccc', linestyle='dashed')
    plt.title(prefix)
    plt.text(0.5, 0.5, plt_label, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.savefig(f"rejection_analysis/{prefix}_reject_curve.png")
    plt.clf()
    
def reject_analysis(GPC: GPClassifier, X_val: np.array, y_val: np.array, thresholds: List[float], sentences: List[str]):

    predictive_mean, predictive_variance = GPC.latent_predictive_distribution(X_val)
    app = GPC.averaged_predictive_probability(predictive_mean, predictive_variance)
    val_accuracies = []
    num_retained = []
    kept_sentences = []
    
    for threshold in thresholds:
        
        confidence = lambda x: np.abs(x - 0.5)
        app_confidence_indices = np.where(confidence(app) > threshold)[0]
        confident_predictions = app[app_confidence_indices]
        confident_y_val = y_val[app_confidence_indices]
        
        if len(confident_predictions) == 0:
            break
        
        kept_sentences.append([sentences[i] for i in app_confidence_indices])
        val_accuracies.append(GPClassifier.classification_accuracy(confident_predictions, confident_y_val))
        num_retained.append(len(app_confidence_indices))
    
    return thresholds[:len(num_retained)], val_accuracies, num_retained, kept_sentences

def train(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, hyperparameter: dict) -> GPClassifier:
    
    GPC = GPClassifier(X_train, y_train, hyperparameter)
    min_NLL = GPC.newton_MAP()
    
    val_predictive_mean, val_predictive_variance = GPC.latent_predictive_distribution(X_val)
    averaged_predictive_probability = GPC.averaged_predictive_probability(val_predictive_mean, val_predictive_variance)  
    validation_accuracy = GPClassifier.classification_accuracy(averaged_predictive_probability, y_val)
    
    train_predictive_mean, train_predictive_variance = GPC.latent_predictive_distribution(X_train)
    averaged_predictive_probability = GPC.averaged_predictive_probability(train_predictive_mean, train_predictive_variance)  
    training_accuracy = GPClassifier.classification_accuracy(averaged_predictive_probability, y_train)
    
    return GPC, validation_accuracy, training_accuracy, min_NLL

def create_training_graphs(results: pd.DataFrame, prefix: str):
        
    sqd_length_scale = results['sqd_length_scale'].to_list()
    signal_variance = results['signal_variance'].to_list()[0]
    validation_accuracy = results['validation_accuracy'].to_list()
    training_accuracy = results['training_accuracy'].to_list()
    NLL = results['NLL'].to_list()
    
    plt.plot(sqd_length_scale, training_accuracy, label='Training Accuracy')
    plt.title(f"{prefix}: Training Accuracy vs. Squared Length Scale (Signal Variance: {signal_variance})")
    plt.xlabel('Squared Length Scale')
    plt.ylabel('Training Accuracy')
    plt.legend(loc='best')
    plt.savefig(f'hyperparam_analysis/{prefix}_sls_training_accuracy.png')
    plt.clf()
    
    plt.plot(sqd_length_scale, validation_accuracy, label='Validation Accuracy')
    plt.title(f"{prefix}: Validation Accuracy vs. Squared Length Scale (Signal Variance: {signal_variance})")
    plt.xlabel('Squared Length Scale')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='best')
    plt.savefig(f'hyperparam_analysis/{prefix}_sls_validation_accuracy.png')
    plt.clf()
    
    plt.plot(sqd_length_scale, NLL, label='Negative Log Likelihood')
    plt.title(f'{prefix}: Negative Log Likelihood vs. Squared Length Scale (Signal Variance: {signal_variance})')
    plt.xlabel('Squared Length Scale')
    plt.ylabel('NLL')
    plt.legend(loc='best')
    plt.savefig(f'hyperparam_analysis/{prefix}_sls_NLL.png')
    plt.clf()
    
def run_training_experiment(prefix: str, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, hyperparameters: List[dict]) -> pd.DataFrame:
    
    results = pd.DataFrame(
        columns=['signal_variance', 
                 'sqd_length_scale', 
                 'training_accuracy', 
                 'validation_accuracy',
                 'NLL',
                 'GPC'])
    
    for i in range(len(hyperparameters)):
        gcp, val_acc, train_acc, NLL = train(X_train, y_train, X_val, y_val, hyperparameters[i])
        logging.info(f"Hyperparameter: {hyperparameters[i]} - Validation Accuracy: {val_acc}")
        row = {
            'signal_variance': hyperparameters[i]['signal_variance'],
            'sqd_length_scale': hyperparameters[i]['sqd_length_scale'],
            'validation_accuracy': val_acc,
            'training_accuracy': train_acc,
            'NLL': NLL,
            'GPC': gcp}
        results = pd.concat([results, pd.DataFrame(row, index=[0])], axis=0, ignore_index=True)
    
    logging.info(f"Training Results for {prefix}: \n {results.to_string()}")
    create_training_graphs(results, prefix)
    return results
        
def run_rejection_analysis_experiment(
        prefix: str, results: pd.DataFrame, X_val: np.array,
        y_val: np.array, val_sentences: List[str]):
    
    # Using best model
    best_row = results[results.validation_accuracy == results.validation_accuracy.max()].iloc[0]
    gpc = best_row['GPC']
    hyperparameters = {
        'signal_variance': best_row['signal_variance'],
        'sqd_length_scale': best_row['sqd_length_scale']}
    thresholds = np.linspace(0,0.5,100)
    
    val_thresholds, val_accuracies, val_num_trained, val_kept_sentences = reject_analysis(
            gpc,
            X_val, 
            y_val, 
            thresholds,
            val_sentences)
    
    plot_reject_curve(
            prefix,
            val_thresholds, 
            val_accuracies, 
            val_num_trained, 
            hyperparameters)
        
    with open(f'rejection_analysis/{prefix}_val_sentences.txt', 'w') as f:
        for i in range(len(val_kept_sentences)):
            f.write(f'{i}: {val_kept_sentences[i]}\n\n') 
         
def load_data(prefix: str):
    
    # training data: inputs, targets, and sentences
    X_train = np.load(prefix+'_X_train.npy')
    y_train = np.load(prefix+'_y_train.npy')
    sentences_train = json.load(open(prefix+'_sentences_train.json','r'))

    # validation data: inputs, targets, and sentences
    X_val = np.load(prefix+'_X_val.npy')
    y_val = np.load(prefix+'_y_val.npy')
    sentences_val = json.load(open(prefix+'_sentences_val.json','r'))

    return X_train, y_train, sentences_train, X_val, y_val, sentences_val

def main():
     # dataset name
    logging.root.setLevel(logging.INFO)
    if not os.path.exists('rejection_analysis'):
        os.mkdir('rejection_analysis')
    if not os.path.exists('hyperparam_analysis'):
        os.mkdir('hyperparam_analysis')
        
    # hyperparameters to consider, as part of your analysis
    all_hyperparams = []
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':.5})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':1})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':2})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':4})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':8})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':16})
    all_hyperparams.append({'signal_variance':10,'sqd_length_scale':32})
    
    X_train, y_train, sentences_train, X_val, y_val, sentences_val = load_data('cola')
    GPC = GPClassifier(X_train, y_train, {'signal_variance':10,'sqd_length_scale':.5})
    GPC.newton_MAP()
    val_predictive_mean, val_predictive_variance = GPC.latent_predictive_distribution(X_val)
    print(val_predictive_variance)
    print(val_predictive_variance.shape)
    print()
    print(val_predictive_mean)
    print(val_predictive_mean.shape)
    print()
    print(X_val)
    print(X_val.shape)
    print()
    print(y_val)
    print(y_val.shape)
    
    # for prefix in ['cola', 'sst2']:
    #     X_train, y_train, sentences_train, X_val, y_val, sentences_val = load_data(prefix)
    #     results = run_training_experiment(prefix, X_train, y_train, X_val, y_val, all_hyperparams)
    #     run_rejection_analysis_experiment(prefix, results, X_val, y_val, sentences_val)
        

    
# Our MAIN
if __name__=='__main__':
    main()
