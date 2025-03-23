"""
Implementing the binary model from the phd thesis of Meghin Spencer, 2017.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from utils import *
from jax.typing import ArrayLike


def sample_orbit(masses: jax.Array, log_g: jax.Array, max_logP: float = 6.51) -> OrbitParams:
    """
    Samples the orbital characteristics of binaries and returns
    the center-of-mass (COM) frame line-of-site velocity (LOSV)
    of the primary for one epoch of observation. Currently using
    the distributions from DM91.
    Args:
        masses: Mass of the primary stars in Msun.
        log_g : log_10(surface gravity) of primary star. Surface gravity in CGS!!!!
    """
    num_stars = masses.shape[0]
    with numpyro.plate('eccentricity', num_stars):
        # Ps & qs
        q = jnp.array(numpyro.sample("mass ratio", dist.TruncatedNormal(loc=0.23, scale=0.42, low=0.1, high=1.0)))
        min_logP = get_min_logP(log_g, masses, q)
        logP = jnp.array(numpyro.sample("logP", dist.TruncatedNormal(loc=4.8, scale=2.3, low=min_logP, high=max_logP)))
        # Eccentricity
        e_low = jnp.array(numpyro.sample("e_low",   dist.Delta(0)))
        e_mid = jnp.array(numpyro.sample("e_mid",   dist.TruncatedNormal(loc=.25, scale=.12, low=0, high=max_ecc(logP))))
        e_high = jnp.array(numpyro.sample("e_high", dist.DoublyTruncatedPowerLaw(1.0, 0, max_ecc(logP))))
        eccentricity = jnp.where(logP <= 1.08, e_low, jnp.where(logP < 3, e_mid, e_high))
        # Euler angles
        mean_anomaly = jnp.array(numpyro.sample('mean anomaly', dist.Uniform(low=0, high=2*jnp.pi)))
        true_anomaly = jax.vmap(get_true_anomaly)(mean_anomaly, eccentricity)
        inclination = jnp.array(numpyro.sample('inclination', OrbitalInclination()))
        periastron = jnp.array(numpyro.sample('periastron', dist.Uniform(low=0, high=2*jnp.pi)))

    params = OrbitParams(m_1          = masses,
                         q            = q,
                         e            = eccentricity,
                         logP         = logP,
                         periapsis    = periastron,
                         inclination  = inclination,
                         mean_anomaly = mean_anomaly,
                         true_anomaly = true_anomaly
                         )
    return params


@jax.jit
def v_r_orb(params: OrbitParams) -> jax.Array:
    """
    Orbital Radial Velocity/ Line-of-sight velocity in the center-of-mass frame of the binary orbit.
    Args:
        params: OrbitParams instance
    Returns:
        v  : Radial velocity of primary star in km/s w.r.t the c.o.m.
    References:
    Implemented as in the PhD Thesis of Meghin Spencer (2017),
    https://deepblue.lib.umich.edu/handle/2027.42/140878.
    """
    G = 132712440000.0  # Newtons constant in km^3/(Msun * s^2)
    P = (10**params.logP)*24*3600  # log_10 days -> seconds
    inner_power = 2*jnp.pi*G*params.m_1/(P*(1+params.q)**2)
    orientation = jnp.sin(params.inclination)*(jnp.cos(params.true_anomaly+params.periapsis)+params.e*jnp.cos(params.periapsis))
    return params.q/jnp.sqrt(1-params.e**2)*inner_power**(1/3)*orientation


def generate_orbit_state(key: jax.Array,
                         masses: ArrayLike,
                         log_g: ArrayLike,
                         ) -> OrbitState:
    """
    Create the initial state of the model.
    Args:
        key   : random key to seed the model with.
        masses: the mass of the primary star in each observation.
        log_g : the surface gravity of the primary star. Used to get minimum period.
    Returns:
        state: Initial state of orbit.
    """
    masses = jnp.asarray(masses)
    log_g  = jnp.asarray(log_g)
    model  = numpyro.handlers.seed(sample_orbit, key)
    params = model(masses, log_g)
    v_r    = v_r_orb(params)
    state = OrbitState(params=params, v_r = v_r)
    return state


@jax.jit
def binary_step(params: OrbitParams, dt_days) -> OrbitState:
    """ Increments the orbits in state by dt_days, saving the new velocities. """
    mean_anomaly = params.mean_anomaly + 2*jnp.pi*dt_days/10**params.logP  # increment orbit by dt in days
    true_anomaly = get_true_anomaly(mean_anomaly, params.e)  # get new orbit phase
    new_params = OrbitParams(m_1=params.m_1, q=params.q, e=params.e, logP=params.logP,
                             periapsis=params.periapsis, inclination=params.inclination,
                             mean_anomaly=mean_anomaly, true_anomaly=true_anomaly)
    v_r = v_r_orb(new_params)
    return OrbitState(params=new_params, v_r=v_r)


def main():
    """ USAGE """
    key = random.key(42)
    num_epochs = 100
    num_stars = 10000
    mass = 0.8
    log_g = 2.0
    binary_fraction = 0.5
    dt = 100
    mean_vlos = 50
    dispersion_vlos = 10
    noise = 0.1
    # We do not need to generate masses/log_g for all of the stars, since it only matters
    # for the binaries. But if we were working in a larger simulation we can assume some binary
    # fraction and do something like this:
    all_masses = jnp.array([mass]*num_stars)
    all_log_g = jnp.array([log_g]*num_stars)
    binary_mask   = random.bernoulli(key=key,p=binary_fraction,shape=all_masses.shape)
    binary_masses = all_masses[binary_mask]
    binary_log_g  = all_log_g[binary_mask]
    # This is us incrementing the orbits of the binary stars num_epochs times, by a step size of dt_days
    # NOTE: masses and log_g dont have to be a jax array, they can be a numpy array etc., but key has to be jax key
    state = generate_orbit_state(key=key,masses=binary_masses,log_g=binary_log_g)
    vr = []
    dt_days=jnp.array([dt]*num_epochs)
    import time
    t0 = time.time()
    for i in range(num_epochs):
        state = binary_step(params=state.params,dt_days=dt_days[i])
        vr.append(state.v_r)
    vr = jnp.array(vr).T # This has shape (num_binaries, num_epochs)
    # If we also want to sample the non-binary orbits this many times, it is much easier. We only
    # need to sample the noise num_epochs times.
    # Sampling the intrinsic line-of-sight velocities for num_stars stars
    v_los = mean_vlos + dispersion_vlos*random.normal(key,shape=all_masses.shape)
    # Now we need to repeat this over the trailing axis
    v_los = jnp.repeat(v_los[:, jnp.newaxis], num_epochs, axis=-1)
    # Ok, now we can sample the noise for num_epochs. Note this is an array of shape (stars, number of epochs)
    # If we have 1000 simulated stars, with 10 epochs each, this has shape (1000, 10)
    v_los_noise = noise*random.normal(key,shape=v_los.shape)
    # We want the binary losv to be expanded to the shape of (stars, number of epochs)
    binary_mask_expanded = jnp.repeat(binary_mask[:, jnp.newaxis], num_epochs, axis=-1)
    print(binary_mask_expanded)
    vr = jnp.place(arr=jnp.zeros(v_los.shape), mask=binary_mask_expanded, vals=vr, inplace=False)
    v_los_total = v_los + v_los_noise + vr
    print(v_los_total)
    t1 = time.time()
    print(t1-t0)

if __name__ == '__main__':
    main()
