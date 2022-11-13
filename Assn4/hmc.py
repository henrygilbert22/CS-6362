from utils import *

from bayesian_matrix_factorization import BayesianMatrixFactorization, _evaluate, _posterior_predictive_update, _evaluate_predictive_mean

import jax
from jax import random
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as python_numpy

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import json

import matplotlib.pyplot as plt

class HMC:
    # Initialize Hamiltonian Monte Carlo with probabilistic model -> we access the log-joint distribution, and its gradient, for HMC
    def __init__(self,probabilistic_model: BayesianMatrixFactorization):
        self.model = probabilistic_model
    #

    # Return potential energy for Hamiltonian
    def potential_energy(self,model_params):
        return -self.model.log_joint_probability(model_params)
    #

    # Return kinetic energy for Hamiltonian
    def kinetic_energy(self,momentum):
        param_vector = np.concatenate([momentum[key].flatten() for key in momentum.keys()])
        return -0.5*param_vector.T@param_vector
    #

    # Return Hamiltonian
    def hamiltonian(self,model_params,momentum):
        pe = self.potential_energy(model_params)
        ke = self.kinetic_energy(momentum)
        return ke+pe
    #

    # Perform leapfrog update -> NOTE: feel free to change the function, and its parameters, as needed for efficient updates
    def leapfrog(self,model_params,momentum,eps):
        
        # Compute gradient of potential energy
        grad_potential_energy = jax.grad(self.potential_energy)(model_params)
        
        # Update momentum
        momentum = {key:momentum[key] + 0.5*eps*grad_potential_energy[key] for key in momentum.keys()}
        
        # Update model parameters
        model_params = {key:model_params[key] + eps*momentum[key] for key in model_params.keys()}
        
        # Update momentum (will require another gradient computation, from updated model parameters)
        momentum = {key:momentum[key] + 0.5*eps*grad_potential_energy[key] for key in momentum.keys()}
        
        return model_params, momentum
    #

    # The main HMC loop: we give test preferences for evaluation, step size (default 2e-3) number of steps (default 50), length of chain (default 1000), and burn-in (default 300)
    def hmc(self,test_prefs,eps=2e-3,L=50,N=1000,burn_in=300, rand_key=1):
        
        # Initialize model parameters
        model_params = create_model_params(self.model.n_users,self.model.n_beers,self.model.D, rand_key)
       
        chain = []
        predictive_mean_sum = 0
        predictive_mean_avg = 0
        
        test_triplet_users = np.array([v[0] for v in test_prefs[0:100]],dtype=np.int32)
        test_triplet_beers = np.array([v[1] for v in test_prefs[0:100]],dtype=np.int32)
        test_triplet_prefs = np.array([v[2] for v in test_prefs[0:100]],dtype=np.float64)
        
        beer_precisions = []
        user_precisions = []
        
        latent_beer_representations = []
        latent_user_representations = []
        
        prediction_averages = []
        prediction_points = []
    
        for i in range(N):
            
            # sample new momentum vector
            momentum = create_model_params(self.model.n_users,self.model.n_beers,self.model.D, rand_key)
            
            # Set initial conditions for integration
            model_params_prime = model_params
            momentum_prime = momentum
            
            for _ in range(L):
                model_params_prime, momentum_prime = self.leapfrog(model_params_prime,momentum_prime,eps)
            
            hamiltonian_prime = self.hamiltonian(model_params_prime, momentum_prime)
            hamiltonian = self.hamiltonian(model_params, momentum)
            
            accpectance_probability = np.exp(hamiltonian_prime/hamiltonian)
            rand_float = random.uniform(advance_rng(rand_key))
            print(f"accpectance_probability: {accpectance_probability} - rand_float: {rand_float}")
            
            if rand_float < accpectance_probability:
                model_params = model_params_prime
                momentum = momentum_prime
                
            if i > burn_in:
                
                chain.append(model_params) 
                beer_precisions.append(model_params['beer_precision'])
                user_precisions.append(model_params['user_precision'])
                latent_beer_representations.append(model_params['latent_beers'])
                latent_user_representations.append(model_params['latent_users'])
                 
                # Online update of predictive mean
                sample_predictive_mean = _posterior_predictive_update(model_params, test_triplet_users, test_triplet_beers)
                predictive_mean_sum += sample_predictive_mean
                predictive_mean_avg = predictive_mean_sum/(i-burn_in) 
                
                prediction_points.append(_evaluate_predictive_mean(sample_predictive_mean, test_triplet_prefs))
                prediction_averages.append(_evaluate_predictive_mean(predictive_mean_avg, test_triplet_prefs))
                
             
        predictive_mean_score = _evaluate_predictive_mean(predictive_mean_avg, test_triplet_prefs)
        return beer_precisions, user_precisions, predictive_mean_score, latent_beer_representations, latent_user_representations, model_params, prediction_points, prediction_averages
    
    #

def precision_trace_plots():
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=10))
    beer_precisions_1, user_precisions_1, _, _, _, _, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    beer_precisions_2, user_precisions_2, _, _, _, _, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=2)
    beer_precisions_3, user_precisions_3, _, _, _, _, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=3)
    beer_precisions_4, user_precisions_4, _, _, _, _, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=4)
    beer_precisions_5, user_precisions_5, _, _, _ , _, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=5)
    
    plt.plot(beer_precisions_1, label='chain 1')
    plt.plot(beer_precisions_2, label='chain 2')
    plt.plot(beer_precisions_3, label='chain 3')
    plt.plot(beer_precisions_4, label='chain 4')
    plt.plot(beer_precisions_5, label='chain 5')
    plt.legend()
    plt.savefig('precision_trace_plots.png')
    

def latent_beer_representation(test_prefs):
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=2))
    _, _, _, latent_beers, latent_users, model_params, _, _ = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    
    saxo = latent_beers[0]
    st_peters_golden_ale = latent_beers[1]
    de_ranke = latent_beers[2]
    scotch_silly = latent_beers[3]
    monts_3 = latent_beers[4]
    
    saxo_dist_draws = python_numpy.random.multivariate_normal(np.mean(saxo, axis=0), model_params["beer_precision"]*np.eye(2), 100)
    st_peters_golden_ale_dist_draws = python_numpy.random.multivariate_normal(np.mean(st_peters_golden_ale, axis=0), model_params["beer_precision"]*np.eye(2), 100)
    de_ranke_dist_draws = python_numpy.random.multivariate_normal(np.mean(de_ranke, axis=0), model_params["beer_precision"]*np.eye(2), 100)
    scotch_silly_dist_draws = python_numpy.random.multivariate_normal(np.mean(scotch_silly, axis=0), model_params["beer_precision"]*np.eye(2), 100)
    monts_3_dist_draws = python_numpy.random.multivariate_normal(np.mean(monts_3, axis=0), model_params["beer_precision"]*np.eye(2), 100)
    
    plt.scatter(*zip(*saxo_dist_draws), label='saxo')
    plt.scatter(*zip(*st_peters_golden_ale_dist_draws), label='st_peters_golden_ale')
    plt.scatter(*zip(*de_ranke_dist_draws), label='de_ranke')
    plt.scatter(*zip(*scotch_silly_dist_draws), label='scotch_silly')
    plt.scatter(*zip(*monts_3_dist_draws), label='monts_3')
    plt.legend()
    plt.savefig('latent_beer_representation.png')

def bayesian_marginalization():
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=2))
    _, _, _, latent_beers, latent_users, model_params, predictive_points_2, predictive_averages_2 = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=4))
    _, _, _, latent_beers, latent_users, model_params, predictive_points_4, predictive_averages_4 = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=8))
    _, _, _, latent_beers, latent_users, model_params, predictive_points_8, predictive_averages_8 = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    
    hmc = HMC(BayesianMatrixFactorization(train_prefs, n_users, n_beers, D=12))
    _, _, _, latent_beers, latent_users, model_params, predictive_points_12, predictive_averages_12 = hmc.hmc(test_prefs, N=10, L=5, burn_in=3, rand_key=1)
    
    plt.plot(predictive_points_12, label='Predicitive Points - 12')
    plt.plot(predictive_points_8, label='Predicitive Points - 8')
    plt.plot(predictive_points_4, label='Predicitive Points - 4')
    plt.plot(predictive_points_2, label='Predicitive Points - 2')
    
    plt.plot(predictive_averages_12, label='Predicitive Averages - 12')
    plt.plot(predictive_averages_8, label='Predicitive Averages - 8')
    plt.plot(predictive_averages_4, label='Predicitive Averages - 4')
    plt.plot(predictive_averages_2, label='Predicitive Averages - 2')
    
    plt.legend()
    plt.savefig('bayesian_marginalization.png')


if __name__=='__main__':
    
    train_prefs = json.load(open('train_prefs.json','r'))
    test_prefs = json.load(open('test_prefs.json','r'))
    n_users = 1+max([v[0] for v in train_prefs])
    n_beers = 1+max([v[1] for v in train_prefs])

    n_users = 100
    n_beers = 100
    
    print('n users',n_users)
    print('n beers',n_beers)
    print('number of training preferences',len(train_prefs))
    
    #precision_trace_plots()
    #latent_beer_representation(test_prefs)
    bayesian_marginalization()

    

