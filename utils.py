import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from typing import NamedTuple


class OrbitParams(NamedTuple):
    """
    Elements:
        Stellar:
        m_1: Mass of primary star in Msun
        q  : Mass ratio of the secondary to primary, defined as m_2/m_1
        e  : Eccentricity of orbit
        P  : Period in days
        Orientation:
        θ  : True Anomaly -> phase of the orbit
        ω  : Argument of periastron/periapsis -> angle between ascending node and periapsis
        i  : inclination
    """
    m_1         : jax.Array 
    q           : jax.Array
    e           : jax.Array
    logP        : jax.Array
    periapsis   : jax.Array
    inclination : jax.Array
    mean_anomaly: jax.Array
    true_anomaly: jax.Array


class OrbitState(NamedTuple):
    """
    Tracks quantities which characterize the binary orbits.
    """
    params: OrbitParams
    v_r: jax.Array


class OrbitalInclination(dist.Distribution):
    support = dist.constraints.interval(0, jnp.pi/2)

    def __init__(self):
        super().__init__(batch_shape=(), event_shape=())

    def sample(self, key, sample_shape=()):
        """
        Sample from the orbital inclination distribution w/ inverse CDF sampling.
        Args:
            key: PRNGKey for random sampling
            sample_shape: Shape of samples to generate
        Returns:
            Samples from orbital inclination distribution in radians
        """
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape)
        return jnp.arccos(1 - u)

    def log_prob(self, value):
        value = jnp.asarray(value)
        if self._validate_args:
            self._validate_sample(value)
        # Clip to [1e-6, pi/2 - 1e-6] to match support
        safe_val = jnp.clip(value, 1e-6, jnp.pi/2 - 1e-6)
        return jnp.log(jnp.sin(safe_val))


def get_true_anomaly(M: jax.Array, e: jax.Array, max_iter=100) -> jax.Array:
    """
    Fixed point iteration to solve for the true anomaly w/ Keplers Equation.
    Args:
        M: mean anomaly
        e: eccentricity
    Returns:
        true_anomaly
    """
    @jax.jit
    def body_fun(carry):
        E_prev, _ = carry
        E_next = M + e * jnp.sin(E_prev)
        return E_next, E_next

    E_init = jnp.ones_like(M) * jnp.pi
    E_final, _ = jax.lax.scan(
        lambda carry, _: (body_fun(carry), None),
        (E_init, E_init),
        xs=None,
        length=max_iter
    )
    E = E_final[0]

    beta = e / (1 + jnp.sqrt(1 - e**2))
    true_anomaly = E + 2 * jnp.arctan(
        beta * jnp.sin(E) / (1 - beta * jnp.cos(E))
    )
    return true_anomaly


@jax.jit
def max_ecc(log_period: jax.Array) -> jax.Array:
    """
    The eccentricity of an orbit is limited by its period. The maximum is given
    as below, where log_period is in units of log_10(days).
    """
    return 1 - (10**log_period/2)**(-2/3)


@jax.jit
def get_min_logP(log_g: jax.Array, masses: jax.Array, q: jax.Array) -> jax.Array:
    """
    We make the approximation that the minimum orbital period corresponds to the
    period assuming the semi-major axis is the radius of the primary star.
    """
    G = 132712440000.0 # Newtons constant in km^3/(Msun * s^2)
    g_km_s2 = jnp.pow(10, log_g)/1e5
    r = jnp.sqrt(G*masses/g_km_s2) # km
    min_P_s = 2*jnp.pi*jnp.sqrt(jnp.pow(r, 3)/(G*masses*(1+q))) # s
    min_P = min_P_s/(3600*24) # s -> days
    return jnp.log10(min_P)
