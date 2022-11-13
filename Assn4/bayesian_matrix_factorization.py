from utils import *

import jax
from jax import random
import jax.numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# A probabilistic model for Bayesian matrix factorization
class BayesianMatrixFactorization:
    def __init__(self,train_prefs,n_users,n_beers,D=10):
        # data and dimensions
        self.train_prefs = train_prefs # preference triples
        self.triplet_users = np.array([v[0] for v in self.train_prefs[0:100]],dtype=np.int32)
        self.triplet_beers = np.array([v[1] for v in self.train_prefs[0:100]],dtype=np.int32)
        self.triplet_prefs = np.array([v[2] for v in self.train_prefs[0:100]],dtype=np.float64)

        self.n_train = self.triplet_prefs.shape[0]
        self.n_users = n_users
        self.n_beers = n_beers
        self.D = D # dimension of user/beer representations

        # hyperparameters - fixed, but can be changed
        self.gamma_scale = 2.0
        self.gamma_shape = 2.0
    #

    # Use JIT below for faster computation -> this is basically a wrapper for the JIT-compiled method
    def log_joint_probability(self,model_params):
        return _log_joint_probability(model_params,self.triplet_users, self.triplet_beers, self.triplet_prefs)
    #

    # return an initialized model
    def get_init_model(self):
        pass
    #
#

#@jax.jit
def _log_joint_probability(model_params, triplet_users, triplet_beers, triplet_prefs):
    
    latent_users = model_params['latent_users']
    latent_beers = model_params['latent_beers']
    user_precision = model_params['user_precision']
    beer_precision = model_params['beer_precision']

    ln_likelihood_mean = np.sum(latent_users[triplet_users]*latent_beers[triplet_beers], axis=1)
    x_ln_likelihood_mean = triplet_prefs - ln_likelihood_mean
    ln_likelihood = -0.5*sum(x_ln_likelihood_mean**2)
    
    # compute determinant for the log latent_user prior
    ln_latent_user_prior = -0.5*user_precision*np.sum(latent_users**2)
    ln_latent_beer_prior = -0.5*beer_precision*np.sum(latent_beers**2)
    
    return (ln_likelihood + ln_latent_user_prior + ln_latent_beer_prior)[0]

# Evaluate a single model, e.g. a single sample from posterior
@jax.jit
def _evaluate(model_params,test_user_ids,test_beer_ids,test_pref_scores):
    return _log_joint_probability(model_params,test_user_ids,test_beer_ids,test_pref_scores)
#

# Evaluate the mean of the predictive distribution
@jax.jit
def _evaluate_predictive_mean(predictive_mean,test_pref_scores):
    return np.sum((predictive_mean - test_pref_scores)**2)
#

# Online update of the mean of the posterior predictive -> should return predictions on withheld test
@jax.jit
def _posterior_predictive_update(posterior_params,test_user_ids,test_beer_ids):
    return posterior_params['latent_users'][test_user_ids,:] @ posterior_params['latent_beers'][test_beer_ids,:].T
#
