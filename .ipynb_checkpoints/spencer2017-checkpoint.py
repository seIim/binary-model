import jax.numpy as jnp, numpyro.distributions as dist, jax.random as random
from jax.typing import ArrayLike
from typing import NamedTuple
import numpyro, jax
from jax import lax


class OrbitalInclination(dist.Distribution):
    support = dist.constraints.interval(0, jnp.pi/2)

    def __init__(self):
        super().__init__(batch_shape = (), event_shape=())

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
        # Clip to [1e-6, π/2 - 1e-6] to match support
        safe_val = jnp.clip(value, 1e-6, jnp.pi/2 - 1e-6)
        return jnp.log(jnp.sin(safe_val))


class OrbitParams(NamedTuple):
    """ Static orbital parameters """
    q           : jax.Array 
    e           : jax.Array 
    logP        : jax.Array 
    periapsis   : jax.Array 
    inclination : jax.Array 
    mean_anomaly: jax.Array 
    true_anomaly: jax.Array 


class VelocityState(NamedTuple):
    """
    Velocity components.
    Params:
        v_r: line-of-site/radial velocity of the primary star in the center of mass reference frame.
        v_i: intrinsic velocity of the stellar system in the galactrocentric frame.
        v_t: B*v_r + v_i <- total observed velocity of the system. B = 1 if binary, else 0.
    """
    v_r: jax.Array
    v_i: jax.Array
    v_t: jax.Array
    v_i_params: jax.Array


class OrbitState(NamedTuple):
    """
    Tracks quantities which characterize the orbits.
    Params:
        velocities: VelocityState
        binaries: Boolean array dictating which stars in the system are binaries.
        params: OrbitParams
    """
    m_1       : jax.Array 
    velocities: VelocityState
    binaries  : jax.Array
    params    : OrbitParams


def get_true_anomaly(M: jax.Array, e: jax.Array, max_iter=100) -> jax.Array:
    """ Fixed point iteration to solve for the true anomaly w/ Keplers Equation.
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
    E_final, _ = lax.scan(
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
def v_r_orb(m_1:jax.Array, params: OrbitParams) -> jax.Array:
    """
    Orbital Radial Velocity/ Line-of-sight velocity in the center-of-mass frame of the binary orbit.
    Args (params elements):
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
    inner_power = 2*jnp.pi*G*m_1/(P*(1+params.q)**2)
    orientation = jnp.sin(params.inclination)*(jnp.cos(params.true_anomaly+params.periapsis)+params.e*jnp.cos(params.periapsis))
    return params.q/jnp.sqrt(1-params.e**2)*inner_power**(1/3)*orientation


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
        q = jnp.array(numpyro.sample("mass ratio", dist.TruncatedNormal(loc=0.23, scale=0.42, low=0.1, high=1.0)))
        min_logP = get_min_logP(log_g, masses, q)
        logP = jnp.array(numpyro.sample("logP",    dist.TruncatedNormal(loc=4.8, scale=2.3, low=min_logP, high=max_logP)))

        e_low = jnp.array(numpyro.sample("e_low",   dist.Delta(0)))
        e_mid = jnp.array(numpyro.sample("e_mid",   dist.TruncatedNormal(loc=.25, scale=.12, low=0, high=max_ecc(logP))))
        e_high = jnp.array(numpyro.sample("e_high", dist.DoublyTruncatedPowerLaw(1.0, 0, max_ecc(logP))))

        eccentricity = jnp.where(logP <= 1.08, e_low, jnp.where(logP < 3, e_mid, e_high))
        mean_anomaly = jnp.array(numpyro.sample('mean anomaly', dist.Uniform(low=0, high=2*jnp.pi)))
        true_anomaly = jax.vmap(get_true_anomaly)(mean_anomaly, eccentricity)
        inclination = jnp.array(numpyro.sample('inclination', OrbitalInclination()))
        periastron = jnp.array(numpyro.sample('periastron', dist.Uniform(low=0, high=2*jnp.pi)))

    params = OrbitParams(q            = q,
                         e            = eccentricity,
                         logP         = logP,
                         periapsis    = periastron,
                         inclination  = inclination,
                         mean_anomaly = mean_anomaly,
                         true_anomaly = true_anomaly
                         )
    return params


def generate_orbit_state(key: jax.Array,
                         bin_frac: float,
                         masses: ArrayLike,
                         log_g: ArrayLike,
                         v_i_loc  = 0,
                         v_i_scale= 1
                         ) -> OrbitState:
    """
    Create the initial state of the model.
    Args:
        key       : random key to seed the model with.
        bin_frac  : binary fraction in the range [0,1].
        masses    : the mass of the primary star in each observation.
        log_g     : the surface gravity of the primary star. Used to get minimum period.
        v_i_loc   : mean of the intrinsic velocity distribution
        v_i_scale : std/dispersion of the intrinsic velocity distribution.
    Returns:
        state: Initial state of type OrbitState.
    """
    masses = jnp.asarray(masses)
    log_g  = jnp.asarray(log_g)
    # jit compile expensive functions
    model  = jax.jit(numpyro.handlers.seed(sample_orbit, key))
    get_vr = jax.jit(v_r_orb)
    # generate a mask determining which stars will be binaries, 
    # then sample the orbital parameters for those stars.
    binaries = random.bernoulli(key=key, p=bin_frac, shape=masses.shape)
    masses_masked = masses[binaries]
    log_g_masked  = log_g[binaries]
    params = model(masses_masked, log_g_masked)
    # calculate the LOSV for each orbit, then expand the v_r array to match
    # the total number of stars. Non-binaries will have a v_r value of 0.
    v_i_params = jnp.array([v_i_loc, v_i_scale])
    v_r = get_vr(masses_masked, params)
    v_r_expanded = jnp.place(arr=jnp.zeros(masses.shape), mask=binaries, vals=v_r, inplace=False)
    v_i = v_i_params[0] + v_i_params[1]*random.normal(key=key, shape=masses.shape) # mu + std * Normal(0,1) -> Normal(mu, std)
    v_t = v_r_expanded + v_i

    v_state = VelocityState(v_r=v_r_expanded,
                            v_i=v_i,
                            v_t=v_t,
                            v_i_params=v_i_params)
    state = OrbitState(m_1=masses,
                       velocities=v_state,
                       binaries=binaries,
                       params=params)
    return state

    
def step(key:jax.Array, state: OrbitState, masses_masked: jax.Array, dt_days: float) -> OrbitState:
    """ Increments the orbits in state by dt_days, saving the new velocities. """
    params = state.params
    binaries = state.binaries
    masses = state.m_1
    # no dynamic array hack
    #masses_masked = jnp.where(binaries, masses, 0)
    #masses_masked = masses_masked[masses_masked > 0]
    
    mean_anomaly = params.mean_anomaly + 2*jnp.pi*dt_days/10**params.logP # increment orbit by dt in days
    true_anomaly = get_true_anomaly(mean_anomaly, params.e)     # get new orbit phase
    
    new_params = OrbitParams(q=params.q, e=params.e, logP=params.logP,
                             periapsis = params.periapsis, inclination = params.inclination,
                             mean_anomaly=mean_anomaly, true_anomaly=true_anomaly)
    
    new_v_r_masked = v_r_orb(masses_masked,new_params)
    new_v_r = jnp.place(arr=jnp.zeros(masses.shape), mask=binaries, vals=new_v_r_masked, inplace=False)

    mu, sd  = state.velocities.v_i_params
    new_v_i = mu + sd*random.normal(key=key, shape=state.m_1.shape)
    
    new_v_state = VelocityState(v_r=new_v_r, v_i=new_v_i,
                                v_t = new_v_r + new_v_i,
                                v_i_params=state.velocities.v_i_params)
    new_state = OrbitState(m_1=state.m_1,
                           velocities=new_v_state,
                           binaries=binaries,
                           params=new_params)
    return new_state


def main():
    key = random.key(11)
    m = jnp.array([0.8]*10_0000)
    log_g = jnp.array([1]*10_0000)
    bin_frac = 0.5
    state = generate_orbit_state(key,bin_frac,m,log_g)
    import matplotlib.pyplot as plt
    plt.hist(jnp.log10(jnp.abs(state.velocities.v_r[state.binaries])), bins=100, density=True)
    plt.xlim(0)
    plt.show()


if __name__ == '__main__':
    main()
