import jax.numpy as jnp, numpyro.distributions as dist, jax.random as random
from typing import NamedTuple
import numpyro, jax, jaxopt
from spencer2017 import *

class VelocityState(NamedTuple):
    """
    Attributes:
        v_intrinsic: velocity in the galactocentric frame.
        v_r_orb    : velocity in the c.o.m frame.
        v_total    : total velocity
    """
    v_intrinsic: jnp.ndarray
    v_r_orb    : jnp.ndarray
    v_total    : jnp.ndarray

class OrbitParams(NamedTuple):
    """
    State of the system.
    """
    m_1  : jnp.ndarray
    q    : jnp.ndarray
    e    : jnp.ndarray
    logP : jnp.ndarray
    i    : jnp.ndarray
    omega: jnp.ndarray
    M    : jnp.ndarray
    theta: jnp.ndarray

class OrbitState(NamedTuple):
    """
    Total state of the model.
    """
    velocity_state: VelocityState
    orbit_params  : OrbitParams
    rng_key       : random.PRNGKey

class OrbitalInclination(dist.Distribution):
    support = dist.constraints.interval(0, jnp.pi/2)

    def __init__(self):
        super().__init__(batch_shape = (), event_shape=())

    def sample(self, key, sample_shape=()):
        """
        Sample from the orbital inclination distribution using inverse CDF sampling.

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
        """
        Compute log probability of the orbital inclination value.

        The PDF is p(i) = sin(i) for i in [0, pi/2]
        Therefore log_prob = log(sin(i))

        Args:
            value: Value to compute probability for

        Returns:
            Log probability of the value
        """
        value = jnp.asarray(value)
        if self._validate_args:
            self._validate_sample(value)

        # Handle edge cases where sin(i) = 0
        safe_val = jnp.clip(value, 1e-6, jnp.pi - 1e-6)
        return jnp.log(jnp.sin(safe_val))

def get_true_anomaly(M,e):
    """
    Numerically compute the true anomaly from Keplers Equation using fixed point
    iteration.
    Args:
        M: Mean anomaly (randomly sampled)
        e: Eccentricity
    """
    @jax.jit
    def kepler_equation_fixed(E):
        return M + e * jnp.sin(E)
    E = jaxopt.FixedPointIteration(fixed_point_fun=kepler_equation_fixed
                                   ).run(init_params=jnp.ones(shape=M.shape)*jnp.pi).params

    beta = e/(1+jnp.sqrt(1-e**2))
    true_anomaly = E + 2*jnp.arctan(beta*jnp.sin(E)/(1-beta*jnp.cos(E)))
    return true_anomaly

@jax.jit
def max_ecc(log_period):
    """
    The eccentricity of an orbit is limited by its period. The maximum is given
    as below, where log_period is in units of log_10(days).
    """
    return 1 - (10**log_period/2)**(-2/3)

def v_r_orb(params: OrbitParams):
    """
    Orbital Radial Velocity/ Line-of-sight velocity in the center-of-mass frame of the binary orbit.
    Args:
        Stellar:
        m_1: Mass of primary star in Msun
        q  : Mass ratio of the secondary to primary, defined as m_2/m_1
        e  : Eccentricity of orbit
        P  : Period in days
        Orientation:
        θ  : True Anomaly -> phase of the orbit
        ω  : Argument of periastron/periapsis -> angle between ascending node and periapsis
        i  : Inclination
    Returns:
        v  : Radial velocity of primary star in km/s w.r.t the c.o.m.
    References:
    Implemented as in the PhD Thesis of Meghin Spencer (2017),
    https://deepblue.lib.umich.edu/handle/2027.42/140878.
    """
    G = 132712440000.0 # Newtons constant in km^3/(Msun * s^2)
    P = (10**params.logP)*24*3600 # log_10 days -> seconds
    inner_power = 2*jnp.pi*G*params.m_1/(P*(1+params.q)**2)
    orientation = jnp.sin(params.i)*(jnp.cos(params.theta+params.omega)+params.e*jnp.cos(params.omega))
    return params.q/jnp.sqrt(1-params.e**2)*inner_power**(1/3)*orientation

class Model:
    """
    Binary orbit model
    """

    def __init__(self,v_galaxy_loc,v_galaxy_scale,masses,binary_fraction):
        self.bin_frac = binary_fraction
        self.v_scale  = v_galaxy_scale
        self.v_loc    = v_galaxy_loc
        self.masses   = masses

    def init(self,key=None) -> OrbitState:
        """
        Generate initial state
        """
        orbit_key, velocity_key = random.split(key)
        self.binaries = dist.Bernoulli(probs=self.bin_frac
                                ).sample(key=key,sample_shape=self.masses.shape)
        binary_masses = self.masses[jnp.argwhere(self.binaries)]
        model = numpyro.handlers.seed(self.sample_orbit, orbit_key)
        orbit_params = model(binary_masses)
        v_i = dist.Normal(loc=self.v_loc, scale=self.v_scale
                                ).sample(key=velocity_key, sample_shape=self.masses.shape)
        v_r = v_r_orb(orbit_params)
        v_r_ = self.binaries.copy().at[jnp.where(self.binaries == 1)[0]].set(v_r)
        v_t = v_i + v_r_
        v_state = VelocityState(v_i,v_r, v_t)
        return OrbitState(v_state, orbit_params, velocity_key)

    def update(self, state: OrbitState, dt: jnp.ndarray) -> OrbitState:
        """
        Implemented before already.
        """
        rng_key, rng_key_step = random.split(state.rng_key)
        state = state.orbit_params
        M = state.M + 2*jnp.pi*dt/10**state.logP # increment orbit by dt in days
        theta = get_true_anomaly(M, state.e)     # get new orbit phase
        v = v_r_orb(state) # get v_r at new phase
        new_orbit_params = OrbitParams(state.m_1, state.q, state.e, state.logP,
                                state.i, state.omega, M, theta) # new state
        new_velocity_state = jnp.array(0)
        new_orbit_state = OrbitState(new_velocity_state, new_orbit_params, rng_key)
        return new_orbit_state

    @staticmethod
    def sample_orbit(masses: jnp.ndarray) -> OrbitParams:
        """
        Samples the orbital characteristics of binaries and returns
        the center-of-mass (COM) frame line-of-site velocity (LOSV)
        of the system for one epoch of observation. Currently using
        the distributions from DM91.
        Args:
            masses: Mass of the primary stars in Msun.
        """
        num_stars = masses.shape[0]
        # Current min/max values are for Leo II. Slightly vary depending on density of galaxay.
        min_logP = 1.57
        max_logP = 6.51

        with numpyro.plate("ps & qs", num_stars):
            q = numpyro.sample("mass ratio", dist.TruncatedNormal(loc=0.23, scale=0.42,
                                                                low=0.1, high=1.0))
            logP = numpyro.sample("logP", dist.TruncatedNormal(loc=4.8, scale=2.3,
                                                            low=min_logP, high=max_logP))

        with numpyro.plate('eccentricity', num_stars):
            eccentricity = jnp.where(
                logP <= 1.08, numpyro.sample("eccentricity_low",
                dist.Delta(0),),
                jnp.where(logP < 3, numpyro.sample("eccentricity_mid",
                dist.TruncatedNormal(loc=.25, scale=.12, low=0, high=max_ecc(logP)),),
                numpyro.sample("eccentricity_high",
                dist.DoublyTruncatedPowerLaw(1.0,0,max_ecc(logP)))
                ))

        M = numpyro.sample('mean anomaly', dist.Uniform(low=0, high=2*jnp.pi), sample_shape=masses.shape)
        true_anomaly = get_true_anomaly(M.flatten(), eccentricity)
        inclination = numpyro.sample('inclination', OrbitalInclination(), sample_shape=masses.shape)
        periastron = numpyro.sample('periastron', dist.Uniform(low=0, high=2*jnp.pi),
                                                            sample_shape=masses.shape)

        params = OrbitParams(masses.flatten(), q.flatten(), eccentricity.flatten(), logP.flatten(),
                            inclination.flatten(), periastron.flatten(), M.flatten(),
                            true_anomaly.flatten())
        return params


def main():
    """Usage"""
    v_mu       = 20  # intrinsic mean velocity
    v_std      = 4   # intrinsic velocity dispersion
    bin_frac   = 0.5 # fraction of binary stars
    masses     = jnp.array([0.8]*10_000) # stellar masses
    model      = Model(v_mu,v_std,masses,bin_frac) # binary model
    key        = random.PRNGKey(42)
    num_epochs = 100 # number of observations
    state = model.init(key) # initialize the model state
    body_fn = jax.jit(model.update)
    for _ in range(num_epochs):
        state = body_fn(state, jnp.array(10))
        # state = model.update(state, jnp.array(10))
    print(state)

def test():
    key = random.PRNGKey(11)
    masses = jnp.ones(shape=(10_000_000,))
    model = Model(binary_fraction=0.5, v_galaxy_loc=20, v_galaxy_scale=5, masses=masses)
    state = model.init(key=key)
    import matplotlib.pyplot as plt
    plt.hist(jnp.log10(jnp.abs(state.velocity_state.v_r_orb + 1e-10)), bins=1000, histtype='step', density=True)
    plt.xlabel(r'$\log_{10}v_\mathrm{r,orb}$')
    plt.ylabel(r'$P(\log_{10}v_\mathrm{r,orb})$')
    plt.xlim(0)
    # plt.hist(state.velocity_state.v_intrinsic, bins=100, histtype='step', label='v_i', density=True)
    plt.legend()
    plt.show()