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
        self.triplet_users = np.array([v[0] for v in self.train_prefs],dtype=np.int32)
        self.triplet_beers = np.array([v[1] for v in self.train_prefs],dtype=np.int32)
        self.triplet_prefs = np.array([v[2] for v in self.train_prefs],dtype=np.float64)

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
        return _log_joint_probability(model_params,self.triplet_users,self.triplet_beers,self.triplet_prefs)
    #

    # return an initialized model
    def get_init_model(self):
        pass
    #
#

@jax.jit
def _log_joint_probability(model_params,triplet_users,triplet_beers,triplet_prefs):
    pass
#

# Evaluate a single model, e.g. a single sample from posterior
@jax.jit
def _evaluate(model_params,test_user_ids,test_beer_ids,test_pref_scores):
    pass
#

# Evaluate the mean of the predictive distribution
@jax.jit
def _evaluate_predictive_mean(predictive_mean,test_pref_scores):
    pass
#

# Online update of the mean of the posterior predictive -> should return predictions on withheld test
@jax.jit
def _posterior_predictive_update(posterior_params,test_user_ids,test_beer_ids):
    pass
#
