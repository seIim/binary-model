import numpyro, jax
import jax.numpy as jnp, numpyro.distributions as dist, jax.random as random
from astropy import constants as c, units as u

# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(1042)


def PiecewisePowerLaw(exponents, x1, xmid, x2, key, num_samples):
    key, sel = random.split(key, 2)
    """
    Th&e mixture coefficient has some jank sign stuff going on
    Need to be able to sample 3-part distributions at some 
    point for excess twin fraction
    """
    a1,a2 = exponents
    a1,a2 = -a1,-a2
    outer_alpha = xmid**(a1-a2)/(a1-1) * (x1**(1-a1)-xmid**(1-a1))
    inner_alpha = (outer_alpha + (xmid**(1-a2) - x2**(1-a2))/(a2-1))
    a1,a2 = -a1,-a2
    first_seg = dist.DoublyTruncatedPowerLaw(a1, x1, xmid)
    secnd_seg = dist.DoublyTruncatedPowerLaw(a2, xmid, x2)
    
    selection = dist.Bernoulli(probs=(1-outer_alpha/inner_alpha)).sample(key=sel, sample_shape=(num_samples,))
    u = random.uniform(key, shape=(num_samples,), minval=0, maxval=1.0)
    
    out = jnp.where(selection == 0, first_seg.icdf(u), secnd_seg.icdf(u))
    return out

@jax.jit
def orbit_angle(eta, eccentricity):
    """
    f . p(eta|e) -> p(phi)
    Transform eccentric anomaly distribution to get correct
    distribution of the 3rd Euler angle, denoted as phi
    """
    return (1 - eccentricity*jnp.cos(eta))/(2*jnp.pi)

@jax.jit
def g_e(eccentricity, theta, psi, phi):
    """
    eccentricity: eccentricity
    theta, psi, phi: 1st, 2nd, 3rd Euler Angles of the orbit
    """
    return jnp.sin(theta)*(jnp.cos(psi-phi)+eccentricity*jnp.cos(phi))

@jax.jit
def log_losv(log_period, eccentricity, theta, psi, phi, mass_ratio, primary_mass):
    """
    Calculates the log_10 line-of-sight velocity of a binary star system
    """
    log_period = jnp.log10(10**log_period/365)
    k = (jnp.abs(g_e(eccentricity, theta, psi, phi)) / jnp.sqrt(1-eccentricity**2))**3 \
          * (2*jnp.pi*mass_ratio)**3 * primary_mass/(1+mass_ratio)**2
    log_losv = (jnp.log10(k) - log_period)/3
    AU_to_km = 1.496e+8
    yr_to_s = 365*24*3600
    log_losv = jnp.log10(10**log_losv * AU_to_km/yr_to_s)
    return log_losv

@jax.jit
def losv(log_period, eccentricity, theta, psi, phi, mass_ratio, primary_mass):
    """
    Calculates the line-of-sight velocity of a binary star system in the com frame
    log_period: log_10 orbit period in days
    primary_mass: mass of primary star in M_\odot
    mass_ratio: M_2/M_1 where M_1 is the primary mass
    """
    G=(c.G.to(u.km**3/(u.Msun * u.s**2))).value
    P = 10**log_period * 3600 * 24
    a = ( G*primary_mass*(1+mass_ratio)/(4*jnp.pi**2) * P**2 )**(1/3)
    g = g_e(eccentricity, theta, psi, phi)
    return 2*jnp.pi*a*g/(P*jnp.sqrt(1-eccentricity**2))

@jax.jit
def max_ecc(log_period):
    return 1 - (10**log_period/2)**(-2/3)

@numpyro.handlers.seed(rng_seed=key)
def binary_model_one_epoch(velocities, masses, binary_frac):
    """ 
    Samples the orbital characteristics of binaries and returns
    the center-of-mass (COM) frame line-of-site velocity (LOSV)
    of the system for one epoch of observation
    """
    num_stars = masses.shape[0]
    with numpyro.plate("stars", num_stars):
        log_period = numpyro.sample("log_period", dist.TruncatedNormal(loc=4.8, scale=2.3,
                                                                       low=0.2, high=10.0)) # log_10 days ?

        eccentricity = jnp.where(
            10**log_period < 11, numpyro.sample("eccentricity_low", dist.Delta(0),),
            jnp.where(10**log_period < 1000, 
                numpyro.sample("eccentricity_mid", dist.TruncatedNormal(loc=.25, scale=.12, 
                                                                        low=0, high=max_ecc(log_period)),
                                ),
                numpyro.sample("eccentricity_high", dist.DoublyTruncatedPowerLaw(0.5,0,max_ecc(log_period)))
            )
        )

        mass_ratio = PiecewisePowerLaw([0.3, -0.5], 0.1, 0.3, 1.0, key, num_stars)
        # mass_ratio = numpyro.sample("mass_ratio", dist.DoublyTruncatedPowerLaw(0.3, 0.1, 1.0,))

        theta = numpyro.sample("theta", dist.Uniform(0, jnp.pi), )
        psi = numpyro.sample("psi", dist.Uniform(0,2*jnp.pi),)
        eta = numpyro.sample("eta", dist.Uniform(0,2*jnp.pi),)
        phi = numpyro.deterministic("phi transform", orbit_angle(eta, eccentricity))
        binaries = numpyro.sample("B", dist.Binomial(probs=binary_frac))
        losvcm = losv(log_period, eccentricity, theta, psi, phi, mass_ratio, masses)
        # log_losvcm = log_losv(log_period, eccentricity, theta, psi, phi, mass_ratio, masses)
        v_combined = velocities + binaries*losvcm
    return v_combined

import matplotlib.pyplot as plt
# masses = jax.random.uniform(key, shape=(1000000,), maxval=0.6) + 0.6
# vs = binary_model_one_epoch(masses, 0.5)
# print(jnp.quantile(vs, q=jnp.array([.15,.5,.85])))