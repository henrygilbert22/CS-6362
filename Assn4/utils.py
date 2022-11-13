import jax
from jax import random
import jax.numpy as np

# pseudorandom number generation -> initialized at zero, but feel free to change for running chains from different starting points


def advance_rng(rand_key):
    prng_key = random.PRNGKey(rand_key)
    prng_key,subkey = random.split(prng_key)
    return subkey
#

# Here we default the parameters of our model, and bundle model parameters into a dictionary
def create_model_params(n_users,n_beers,D, rand_key):
    model_params = dict()
    init_prec = D
    model_params['latent_users'] = random.normal(advance_rng(rand_key),(n_users,D),dtype=np.float64) / np.sqrt(init_prec)
    model_params['latent_beers'] = random.normal(advance_rng(rand_key),(n_beers,D),dtype=np.float64) / np.sqrt(init_prec)
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
    return {
        'latent_users': random.normal(advance_rng(),model_params['latent_users'].shape,dtype=np.float64),
        'latent_beers': random.normal(advance_rng(),model_params['latent_beers'].shape,dtype=np.float64),
        'user_precision': random.normal(advance_rng(),model_params['user_precision'].shape,dtype=np.float64),
        'beer_precision': random.normal(advance_rng(),model_params['beer_precision'].shape,dtype=np.float64)
    }
#
