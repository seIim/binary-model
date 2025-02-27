"""
Optimizing the detection fraction of binaries in mock samples of dwarf galaxies.
"""
import jax, jax.numpy as jnp, jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, SA
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from functools import partial
from new import *


@partial(jax.jit, static_argnums=(2,3))
def compute_vr(state, dt_days, num_epochs, n_binaries):
    def body_fn(i, state_vr):
        state, vr = state_vr
        state = binary_step(params=state.params, dt_days=dt_days[i])
        vr = vr.at[:, i].set(state.v_r)
        return state, vr

    vr_shape = (n_binaries, num_epochs)
    vr = jnp.zeros(vr_shape)

    final_state, vr = jax.lax.fori_loop(0, num_epochs, body_fn, (state, vr))

    return vr


@partial(jax.jit, static_argnums=(2,6,8))
def generate_velocities(state: OrbitState, dt_days: jax.Array, num_epochs: int,
                        mean_vlos, dispersion_vlos, obs_noise, n_binaries, key,
                        n_stars, binary_mask
                        ):
    
    vr = compute_vr(state, dt_days, num_epochs, n_binaries)
    v_los = mean_vlos + dispersion_vlos*random.normal(
                            key, shape=(n_stars,))
    v_los = jnp.repeat(v_los[:, jnp.newaxis], num_epochs, axis=-1)
    v_los_noise = obs_noise*random.normal(key,shape=v_los.shape)
    binary_mask_expanded = jnp.repeat(binary_mask[:, jnp.newaxis],
                                      num_epochs, axis=-1)
    vr = jnp.place(arr=jnp.zeros(v_los.shape),
                   mask=binary_mask_expanded,
                   vals=vr, inplace=False)
    v_los_total = v_los + v_los_noise + vr
    return v_los_total


def calculate_log_p_H0(observed_velocities, sigma):
    """Calculate log likelihood under H0 (single star) for each star."""
    n = observed_velocities.shape[1]
    sample_mean = jnp.mean(observed_velocities, axis=1)
    sample_var = jnp.var(observed_velocities, axis=1, ddof=1)
    term1 = - (n - 1)/2 * jnp.log(2 * jnp.pi)
    term2 = - (n - 1) * jnp.log(sigma)
    term3 = - 0.5 * jnp.log(n)
    term4 = - (n - 1) * sample_var / (2 * sigma**2)
    log_p_H0 = term1 + term2 + term3 + term4
    return log_p_H0

def calculate_log_p_H1(observed_velocities, true_binary_sequences, sigma):
    """Log likelihood under H1 (binary star)."""
    M = true_binary_sequences.shape[0]
    differences = observed_velocities[:, jnp.newaxis, :] - true_binary_sequences[jnp.newaxis, :, :]
    sum_sq = jnp.sum((differences / sigma)**2, axis=2)  # Shape: (2000, M)
    log_likelihoods = -0.5 * sum_sq - observed_velocities.shape[1] * jnp.log(sigma) - 0.5 * observed_velocities.shape[1] * jnp.log(2 * jnp.pi)
    log_p_H1 = logsumexp(log_likelihoods, axis=1) - jnp.log(M)
    return log_p_H1

def run_experiment(obs_state, model_state, params):
    """Run one experiment for a given parameter set."""
    mean_vlos, dispersion_vlos, num_epochs, dt_day, obs_noise, threshold, n_binaries, n_stars, key, binary_mask = params
    model_key, obs_key = random.split(key)
    v_t_mod = generate_velocities(model_state, dt_days=dt_day, num_epochs=num_epochs,
                                  mean_vlos=0.0,
                                  dispersion_vlos=1.0,
                                  obs_noise=0.0,
                                  n_binaries=n_stars,
                                  n_stars=n_stars,
                                  key=model_key,
                                  binary_mask=binary_mask
                                  )
    v_t_obs = generate_velocities(obs_state, dt_days=dt_day, num_epochs=num_epochs,
                                  mean_vlos=0.0,
                                  dispersion_vlos=1.0,
                                  obs_noise=obs_noise,
                                  n_binaries=n_binaries,
                                  n_stars=n_stars,
                                  key=obs_key,
                                  binary_mask=binary_mask
                                  ) 
    log_p_H0 = calculate_log_p_H0(v_t_obs, obs_noise)
    log_p_H1 = calculate_log_p_H1(v_t_obs, v_t_mod, obs_noise)
   
    log_odds = log_p_H1 - log_p_H0
    posterior_prob_H1 = 1 / (1 + jnp.exp(-log_odds))
    predicted = posterior_prob_H1 > threshold

    TP = jnp.sum((predicted == 1) & (binary_mask == 1))
    return (TP / binary_mask.sum()).astype(float)
    #return TP.item()/binary_mask.sum().item()


def run_experiment_disp(obs_state, model_state, params):
    """Run one experiment for a given parameter set."""
    mean_vlos, dispersion_vlos, num_epochs, dt_day, obs_noise, threshold, n_binaries, n_stars, key, binary_mask = params
    model_key, obs_key = random.split(key)
    v_t_mod = generate_velocities(model_state, dt_days=dt_day, num_epochs=num_epochs,
                                  mean_vlos=0.0,
                                  dispersion_vlos=1.0,
                                  obs_noise=0.0,
                                  n_binaries=n_stars,
                                  n_stars=n_stars,
                                  key=model_key,
                                  binary_mask=binary_mask
                                  )
    v_t_obs = generate_velocities(obs_state, dt_days=dt_day, num_epochs=num_epochs,
                                  mean_vlos=0.0,
                                  dispersion_vlos=1.0,
                                  obs_noise=obs_noise,
                                  n_binaries=n_binaries,
                                  n_stars=n_stars,
                                  key=obs_key,
                                  binary_mask=binary_mask
                                  ) 
    log_p_H0 = calculate_log_p_H0(v_t_obs, obs_noise)
    log_p_H1 = calculate_log_p_H1(v_t_obs, v_t_mod, obs_noise)
   
    log_odds = log_p_H1 - log_p_H0
    posterior_prob_H1 = 1 / (1 + jnp.exp(-log_odds))
    predicted = posterior_prob_H1 > threshold
    
    disp = v_t_obs[~predicted].std()
    return disp 


def main(noise):
    key = random.key(42)
    mass = 0.8
    num_stars = 1000
    log_g = 2.0
    binary_fraction = 0.5
    mean_vlos = 0.0
    dispersion_vlos = 1.0
    obs_noise = noise

    mask_key, grid_key, obs_key = random.split(key, 3)
    # generate masses and log_gs statically 
    all_masses = jnp.array([mass]*num_stars)
    all_log_g = jnp.array([log_g]*num_stars)
    # select binary stars out of all stars
    binary_mask   = random.bernoulli(key=key,
                                     p=binary_fraction,
                                     shape=all_masses.shape)
    binary_masses = all_masses[binary_mask]
    binary_log_g  = all_log_g[binary_mask]
    
    # sample initial orbital parameters for a model grid and observations seperately
    model_state = generate_orbit_state(key=grid_key,
                                      masses=all_masses,
                                      log_g=all_log_g
                                       )
    obs_state = generate_orbit_state(obs_key, 
                                    masses=binary_masses,
                                    log_g=binary_log_g
                                     )
            
    def model(obs_state, model_state, params):
        mean_vlos, dispersion_vlos, num_epochs, _, obs_noise, threshold, n_binaries, n_stars, key, binary_mask = params
        total_time = numpyro.sample("total_time", dist.Uniform(0.0, 2*365.25))
        proportions = numpyro.sample("proportions", dist.Dirichlet(jnp.ones(num_epochs)))
        dt_days_proposal = total_time * proportions
        detection_fraction = objective_fn(dt_days_proposal, obs_state, model_state, params)
        detection_fraction_float = numpyro.deterministic("detection_fraction", detection_fraction.astype(jnp.float32))
        numpyro.factor("obs", detection_fraction_float)

    @jax.jit
    def objective_fn(dt_days, obs_state, model_state, params):
        new_params = (mean_vlos, dispersion_vlos, num_epochs, dt_days, obs_noise, threshold, n_binaries, n_stars, key, binary_mask)
        return run_experiment(obs_state, model_state, new_params)

    def get_disp(dt_days, obs_state, model_state, params):
        new_params = (mean_vlos, dispersion_vlos, num_epochs, dt_days, obs_noise, threshold, n_binaries, n_stars, key, binary_mask)
        return run_experiment_disp(obs_state, model_state, new_params)

    ep, tp, best_tp, disp, best_disp = [], [], [], [], []
    for i in range(1,6):
        num_epochs = i
        n_binaries = obs_state.v_r.shape[0]
        n_stars = all_masses.shape[0]
        dt_day = jnp.array([2*365.25/num_epochs]*num_epochs)
        threshold=0.9997
        params = (mean_vlos, dispersion_vlos, num_epochs, dt_day, obs_noise, threshold, n_binaries, n_stars, key, binary_mask)
        sampler = numpyro.infer.AIES(model=partial(model, obs_state, model_state, params))
        mcmc = MCMC(sampler, num_warmup=500, num_samples=100,
                    num_chains=20,
                    chain_method='vectorized')
        mcmc.run(random.key(0),
                 )
        samples = mcmc.get_samples()
        dt_days_samples = samples["total_time"][:, None] * samples["proportions"]
        res_probs = jax.vmap(lambda dt: objective_fn(dt, obs_state, model_state, params))(dt_days_samples)
        best_idx = jnp.argmax(res_probs)
        best_dt_days = dt_days_samples[best_idx]
        ep.append(num_epochs)
        tp.append(objective_fn(dt_day, obs_state, model_state, params))
        best_tp.append(objective_fn(best_dt_days, obs_state, model_state, params))
        disp.append(get_disp(dt_day, obs_state, model_state, params))
        best_disp.append(get_disp(best_dt_days, obs_state, model_state, params))

    return ep, tp, best_tp, disp, best_disp

#main()
