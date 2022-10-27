import jax
from jax import random
import jax.numpy as np

# pseudorandom number generation -> initialized at zero, but feel free to change for running chains from different starting points
prng_key = random.PRNGKey(0)

def advance_rng():
    global prng_key
    prng_key,subkey = random.split(prng_key)
    return subkey
#

# Here we default the parameters of our model, and bundle model parameters into a dictionary
def create_model_params(n_users,n_beers,D):
    model_params = dict()
    init_prec = D
    model_params['latent_users'] = random.normal(advance_rng(),(n_users,D),dtype=np.float64) / np.sqrt(init_prec)
    model_params['latent_beers'] = random.normal(advance_rng(),(n_beers,D),dtype=np.float64) / np.sqrt(init_prec)
    model_params['user_precision'] = np.log(init_prec*np.ones(1,dtype=np.float64))
    model_params['beer_precision'] = np.log(init_prec*np.ones(1,dtype=np.float64))
    return model_params
#

# You might find it useful to return an initialized set of model parameter gradients (wrt log joint distribution)
def initialize_model_param_grads(model_params):
    pass
#

# Draw momentum - in 1:1 correspondence with model parameters
def draw_momentum(model_params):
    pass
#
