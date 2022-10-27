from utils import *

from bayesian_matrix_factorization import BayesianMatrixFactorization, _evaluate, _posterior_predictive_update, _evaluate_predictive_mean

import jax
from jax import random
import jax.numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import json

import matplotlib.pyplot as plt

class HMC:
    # Initialize Hamiltonian Monte Carlo with probabilistic model -> we access the log-joint distribution, and its gradient, for HMC
    def __init__(self,probabilistic_model):
        self.model = probabilistic_model
    #

    # Return potential energy for Hamiltonian
    def potential_energy(self,model_params):
        pass
    #

    # Return kinetic energy for Hamiltonian
    def kinetic_energy(self,momentum):
        pass
    #

    # Return Hamiltonian
    def hamiltonian(self,model_params,momentum):
        pass
    #

    # Perform leapfrog update -> NOTE: feel free to change the function, and its parameters, as needed for efficient updates
    def leapfrog(self,model_params,momentum,eps):
        pass
    #

    # The main HMC loop: we give test preferences for evaluation, step size (default 2e-3) number of steps (default 50), length of chain (default 1000), and burn-in (default 300)
    def hmc(self,test_prefs,eps=2e-3,L=50,N=1000,burn_in=300):
        pass
    #

if __name__=='__main__':
    train_prefs = json.load(open('train_prefs.json','r'))
    test_prefs = json.load(open('test_prefs.json','r'))
    n_users = 1+max([v[0] for v in train_prefs])
    n_beers = 1+max([v[1] for v in train_prefs])

    print('n users',n_users)
    print('n beers',n_beers)
    print('number of training preferences',len(train_prefs))

    # TODO: entry point for (1) building probabilistic model, and (2) running HMC
#
