"""
Optimizing the detection fraction of binaries
in mock samples of dwarf galaxies.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, SA
from jax.scipy.special import logsumexp
from functools import partial
from spencer2017 import *


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


def generate_velocities(state: OrbitState, dt_days: jax.Array, num_epochs: int,
                        mean_vlos, dispersion_vlos, obs_noise, n_binaries, key,
                        n_stars, binary_mask
                        ):
    vr = compute_vr(state, dt_days, num_epochs, n_binaries)
    v_los = mean_vlos + dispersion_vlos*random.normal(
                            key, shape=(n_stars,))
    v_los = jnp.repeat(v_los[:, jnp.newaxis], num_epochs, axis=-1)
    v_los_noise = obs_noise*random.normal(key, shape=v_los.shape)
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
    # sample_mean = jnp.mean(observed_velocities, axis=1)
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
    mean_vlos, dispersion_vlos, num_epochs, dt_day, obs_noise, threshold, n_obs_binaries, n_obs_stars, key, obs_binary_mask = params
    model_key, obs_key = random.split(key)
    
    # Generate model velocities (all binaries)
    model_n_binaries = model_state.params.m_1.shape[0]
    model_binary_mask = jnp.ones(model_n_binaries, dtype=bool)
    v_t_mod = generate_velocities(
        model_state,
        dt_days=dt_day,
        num_epochs=num_epochs,
        mean_vlos=0.0,
        dispersion_vlos=1.0,
        obs_noise=0.0,
        n_binaries=model_n_binaries,
        n_stars=model_n_binaries,
        key=model_key,
        binary_mask=model_binary_mask
    )
    
    # Generate observed velocities
    v_t_obs = generate_velocities(
        obs_state,
        dt_days=dt_day,
        num_epochs=num_epochs,
        mean_vlos=0.0,
        dispersion_vlos=1.0,
        obs_noise=obs_noise,
        n_binaries=n_obs_binaries,
        n_stars=n_obs_stars,
        key=obs_key,
        binary_mask=obs_binary_mask
    )
    
    log_p_H0 = calculate_log_p_H0(v_t_obs, obs_noise)
    log_p_H1 = calculate_log_p_H1(v_t_obs, v_t_mod, obs_noise)

    log_odds = log_p_H1 - log_p_H0
    posterior_prob_H1 = 1 / (1 + jnp.exp(-log_odds))
    predicted = posterior_prob_H1 > threshold

    TP = jnp.sum((predicted == 1) & (obs_binary_mask == 1))
    return (TP / obs_binary_mask.sum()).astype(float)


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
    """MCMC"""
    key = random.key(42)
    mass = 0.8
    num_obs_stars = 1000
    model_n_binaries = 10000
    log_g = 2.0
    binary_fraction = 0.5
    mean_vlos = 0.0
    dispersion_vlos = 1.0
    obs_noise = noise

    model_key, grid_key, obs_key = random.split(key, 3)
    model_masses = jnp.array([mass] * model_n_binaries)
    model_log_g = jnp.array([log_g] * model_n_binaries)
    model_state = generate_orbit_state(key=grid_key, masses=model_masses, log_g=model_log_g)

    mask_key, obs_key = random.split(obs_key)
    all_masses = jnp.array([mass] * num_obs_stars)
    all_log_g = jnp.array([log_g] * num_obs_stars)
    binary_mask = random.bernoulli(key=mask_key, p=binary_fraction, shape=all_masses.shape)
    binary_masses = all_masses[binary_mask]
    binary_log_g = all_log_g[binary_mask]
    obs_state = generate_orbit_state(obs_key, masses=binary_masses, log_g=binary_log_g)
    
    n_obs_binaries = binary_mask.sum()
    n_obs_stars = num_obs_stars

    run_exp_partial = partial(run_experiment, obs_state, model_state)
    run_disp_partial = partial(run_experiment_disp, obs_state, model_state)

    def objective_fn(dt_days, num_epochs, threshold):
        new_params = (
            mean_vlos, dispersion_vlos, num_epochs, dt_days,
            obs_noise, threshold, n_obs_binaries, n_obs_stars, key, binary_mask
        )
        return run_exp_partial(new_params)

    def get_disp(dt_days, num_epochs, threshold):
        new_params = (
            mean_vlos, dispersion_vlos, num_epochs, dt_days,
            obs_noise, threshold, n_obs_binaries, n_obs_stars, key, binary_mask
        )
        return run_disp_partial(new_params)

    ep, tp, best_tp, disp, best_disp = [], [], [], [], []
    for i in range(1, 6):
        num_epochs = i
        threshold = 0.9997
        dt_day = jnp.array([2*365.25/num_epochs]*num_epochs)

        def model():
            total_time = numpyro.sample("total_time", dist.Uniform(0.0, 2*365.25))
            proportions = numpyro.sample("proportions", dist.Dirichlet(jnp.ones(num_epochs)))
            dt_days_proposal = total_time * proportions
            detection_fraction = objective_fn(dt_days_proposal, num_epochs, threshold)
            numpyro.deterministic("detection_fraction", detection_fraction)
            numpyro.factor("obs", detection_fraction)

        # Run MCMC
        sampler = numpyro.infer.AIES(model)
        mcmc = numpyro.infer.MCMC(
            sampler,
            num_warmup=1000,
            num_samples=200,
            num_chains=50,
            chain_method='vectorized'
        )
        mcmc.run(random.key(0))

        samples = mcmc.get_samples()
        dt_days_samples = samples["total_time"][:, None] * samples["proportions"]
        res_probs = jax.vmap(lambda dt: objective_fn(dt, num_epochs, threshold))(dt_days_samples)
        best_idx = jnp.argmax(res_probs)
        best_dt_days = dt_days_samples[best_idx]

        ep.append(num_epochs)
        tp.append(objective_fn(dt_day, num_epochs, threshold))
        best_tp.append(objective_fn(best_dt_days, num_epochs, threshold))
        disp.append(get_disp(dt_day, num_epochs, threshold))
        best_disp.append(get_disp(best_dt_days, num_epochs, threshold))

    return ep, tp, best_tp, disp, best_disp


def dispersion_reduction_experiment(num_stars=1000,
                                   binary_fraction=0.5,
                                   obs_noise=0.1,
                                   threshold=0.9997,
                                   max_epochs=10):
    """
    Measures how excluding detected binaries reduces observed velocity dispersion
    as the number of observational epochs increases.
    """
    key = random.key(11)
    mass = 0.8
    log_g = 2.0
    true_dispersion = 1.0
    
    model_key, data_key = random.split(key)
    model_masses = jnp.array([mass] * 100000)
    model_log_g = jnp.array([log_g] * 100000)
    model_state = generate_orbit_state(model_key, model_masses, model_log_g)

    data_key, bin_key, vel_key = random.split(data_key, 3)
    binary_mask = random.bernoulli(bin_key, binary_fraction, (num_stars,))
    
    single_vlos = true_dispersion * random.normal(vel_key, (num_stars,))
    
    binary_masses = jnp.where(binary_mask, mass, 0.0)
    binary_log_gs = jnp.where(binary_mask, log_g, 0.0)
    obs_state = generate_orbit_state(vel_key, binary_masses[binary_mask], 
                                   binary_log_gs[binary_mask])

    results = []
    
    for num_epochs in range(1, max_epochs+1):
        key, epoch_key = random.split(key)
        dt_days = jnp.array([365.25/num_epochs]*num_epochs)

        model_vr = compute_vr(
            model_state,
            dt_days=dt_days,
            num_epochs=num_epochs,
            n_binaries=model_state.v_r.shape[0]
        )

        obs_vr = compute_vr(
            obs_state,
            dt_days=dt_days,
            num_epochs=num_epochs,
            n_binaries=binary_mask.sum()
        )
        
        all_vr = jnp.zeros((num_stars, num_epochs))
        all_vr = all_vr.at[binary_mask].set(obs_vr)
        noise = obs_noise * random.normal(epoch_key, (num_stars, num_epochs))
        observed_vlos = true_dispersion * random.normal(vel_key, (num_stars,))[:, None] 
        observed_vlos += noise + all_vr

        log_p_H0 = calculate_log_p_H0(observed_vlos, obs_noise)
        log_p_H1 = calculate_log_p_H1(observed_vlos, model_vr, obs_noise)  # Fixed shape
        
        log_odds = log_p_H1 - log_p_H0
        posterior_prob = 1 / (1 + jnp.exp(-log_odds))
        detected = posterior_prob > threshold

        clean_vlos = jnp.mean(observed_vlos[~detected], axis=1)  # Average over epochs
        measured_disp = jnp.std(clean_vlos) if clean_vlos.size > 10 else jnp.nan
        
        results.append((num_epochs, measured_disp / true_dispersion))

    epochs, ratios = zip(*results)
    return jnp.array(epochs), jnp.array(ratios)


def dispersion_reduction_with_errors(num_stars=1000,
                                    binary_fraction=0.5,
                                    obs_noise=0.1,
                                    threshold=0.9997,
                                    max_epochs=10,
                                    num_bootstraps=100):
    """
    Measures dispersion reduction with uncertainties using bootstrap resampling
    without vectorization for memory efficiency
    """
    key = random.key(11)
    true_dispersion = 1.0
    
    model_key, data_key = random.split(key)
    model_state = generate_orbit_state(
        model_key, 
        masses=jnp.full(100000, 0.8),
        log_g=jnp.full(100000, 2.0)
    )

    data_key, bin_key = random.split(data_key)
    binary_mask = random.bernoulli(bin_key, binary_fraction, (num_stars,))
    obs_state = generate_orbit_state(
        data_key,
        masses=jnp.where(binary_mask, 0.8, 0.0)[binary_mask],
        log_g=jnp.where(binary_mask, 2.0, 0.0)[binary_mask]
    )

    results = {i: [] for i in range(1, max_epochs+1)}
    
    def process_epoch(num_epochs, key):
        """Process single epoch count with given random key"""
        # Split keys
        dt_key, noise_key, sample_key = random.split(key, 3)
        
        # Generate velocities
        dt_days = jnp.array([365.25/num_epochs]*num_epochs)
        
        # Model velocities
        model_vr = compute_vr(
            model_state,
            dt_days=dt_days,
            num_epochs=num_epochs,
            n_binaries=100000
        )
        
        # Observed velocities
        obs_vr = compute_vr(
            obs_state,
            dt_days=dt_days,
            num_epochs=num_epochs,
            n_binaries=binary_mask.sum()
        )
        binary_vr = jnp.zeros((num_stars, num_epochs))
        binary_vr = binary_vr.at[binary_mask].set(obs_vr)
        
        # Generate new noise and single-star velocities
        single_vlos = true_dispersion * random.normal(noise_key, (num_stars,))
        noise = obs_noise * random.normal(noise_key, (num_stars, num_epochs))
        observed_vlos = single_vlos[:, None] + noise + binary_vr

        sample_idx = random.choice(sample_key, num_stars, (num_stars,))
        observed_sample = observed_vlos[sample_idx]

        log_p_H0 = calculate_log_p_H0(observed_sample, obs_noise)
        log_p_H1 = calculate_log_p_H1(observed_sample, model_vr, obs_noise)
        
        log_odds = log_p_H1 - log_p_H0
        posterior_prob = 1 / (1 + jnp.exp(-log_odds))
        detected = posterior_prob > threshold

        clean_vlos = jnp.mean(observed_sample[~detected], axis=1)
        return jnp.std(clean_vlos) / true_dispersion

    keys = random.split(key, max_epochs * num_bootstraps)
    
    for epoch in range(1, max_epochs+1):
        epoch_keys = keys[(epoch-1)*num_bootstraps : epoch*num_bootstraps]
        
        for bootstrap_idx in range(num_bootstraps):
            ratio = process_epoch(epoch, epoch_keys[bootstrap_idx])
            results[epoch].append(ratio)
            
        print(f"Completed epoch {epoch}/{max_epochs}")

    epochs = list(range(1, max_epochs+1))
    mean_ratios = jnp.array([jnp.nanmean(jnp.asarray(results[e])) for e in epochs])
    std_errors = jnp.array([jnp.nanstd(jnp.asarray(results[e]))/jnp.sqrt(num_bootstraps) for e in epochs])
    
    return epochs, mean_ratios, std_errors


def precomputed_dispersion_experiment(num_stars=100,
                                      binary_fraction=0.5,
                                      obs_noise=0.1,
                                      max_epochs=10):
    """
    Measure dispersion reduction using precomputed consistent velocity curves
    Returns: epochs, dispersion_ratios
    """
    key = random.key(11)
    true_dispersion = 1.0
    model_grid_size = 1000000  # Number of binary templates in model grid

    # Generate observed sample -----------------------------------------------
    key, bin_key, orb_key, noise_key = random.split(key, 4)

    # 1. Binary mask for observed stars
    binary_mask = random.bernoulli(bin_key, binary_fraction, (num_stars,))
    num_binaries = binary_mask.sum()

    # 2. Generate persistent binary orbits (all epochs)
    dt_days = jnp.array([365.25/max_epochs]*max_epochs)  # Annual observations
    binary_state = generate_orbit_state(
        orb_key,
        masses=jnp.full(num_binaries, 0.8),
        log_g=jnp.full(num_binaries, 2.0)
    )
    binary_vr = compute_vr(binary_state, dt_days, max_epochs, num_binaries)

    # 3. Generate persistent single-star velocities + noise
    single_vlos = true_dispersion * random.normal(noise_key, (num_stars,))
    noise = obs_noise * random.normal(noise_key, (num_stars, max_epochs))

    # 4. Combine into full velocity array (shape: stars × epochs)
    all_velocities = jnp.zeros((num_stars, max_epochs))
    all_velocities = all_velocities.at[binary_mask].add(binary_vr)
    all_velocities += single_vlos[:, None] + noise

    # Precompute model grid ---------------------------------------------------
    key, model_key = random.split(key)
    model_state = generate_orbit_state(
        model_key,
        masses=jnp.full(model_grid_size, 0.8),
        log_g=jnp.full(model_grid_size, 2.0)
    )
    model_vr = compute_vr(model_state, dt_days, max_epochs, model_grid_size)

    # Calculate dispersion ratios ---------------------------------------------
    ratios = []

    for num_epochs in range(1, max_epochs+1):
        # Use first N epochs for both observed and model
        observed = all_velocities[:, :num_epochs]
        models = model_vr[:, :num_epochs]

        # Detection calculation
        log_p_H0 = calculate_log_p_H0(observed, obs_noise)
        log_p_H1 = calculate_log_p_H1(observed, models, obs_noise)

        # Binary detection
        log_odds = log_p_H1 - log_p_H0
        posterior = 1 / (1 + jnp.exp(-log_odds))
        detected = posterior > 0.9997

        clean_vlos = jnp.mean(observed[~detected], axis=1)
        valid = clean_vlos.size > 10  # Require minimum sample size
        ratio = jnp.where(valid, jnp.std(clean_vlos)/true_dispersion, jnp.nan)
        ratios.append(ratio)

    return (jnp.arange(1, max_epochs+1),
            jnp.array(ratios))


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    for i in [0.1, 0.5, 1.0]:
        epochs, ratios = precomputed_dispersion_experiment(
            num_stars=100,
            binary_fraction=0.5,
            obs_noise=i,
            max_epochs=10
        )

        plt.plot(epochs, ratios, 'o-', label=i)
        plt.xlabel("Epochs")
        plt.ylabel(r"$\mathrm{\sigma/\sigma_0}$")
        plt.legend()

    plt.show()
