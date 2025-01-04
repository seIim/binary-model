import jax.numpy as jnp, numpyro.distributions as dist, jax.random as random
from astropy import constants as c, units as u
import numpyro, jax, jaxopt
from typing import NamedTuple


key = random.PRNGKey(42)

class OrbitParams(NamedTuple):
    m_1  : float
    q    : float
    e    : float
    P    : float
    i    : float
    omega: float
    v    : float
    M    : float
    theta: float

class OrbitState(NamedTuple):
    params: OrbitParams

    def init(self) -> None:
        v = v_r_orb(jnp.array(self.params.m_1), self.params.q,
                                self.params.e, jnp.log10(self.params.P),
                                self.params.theta, self.params.omega,
                                self.params.i)
        return OrbitState(OrbitParams(self.params.m_1,self.params.q,
                                      self.params.e, self.params.P,
                                      self.params.i, self.params.omega,
                                      v, self.params.M, self.params.theta))

    def update(self, dt) -> NamedTuple:
        M = jnp.array(self.params.M + 2*jnp.pi*dt/self.params.P)
        theta = get_true_anomaly(M, self.params.e)
        v = v_r_orb(self.params.m_1, self.params.q, self.params.e,
                    jnp.log10(self.params.P), theta, self.params.omega,
                    self.params.i)
        params = OrbitParams(self.params.m_1,self.params.q,
                    self.params.e, self.params.P,
                    self.params.i, self.params.omega,
                    v, M, theta)
        return OrbitState(params)

def inclination_dist(key, shape):
    icdf = lambda u: jnp.arccos(1 - u)
    u = random.uniform(key, shape)
    return icdf(u)

def get_true_anomaly(M,e):
    """
    Numerically compute the true anomaly from Keplers Equation using fixed point
    iteration.
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
    as below, where log_period is given in log_10(days).
    """
    return 1 - (10**log_period/2)**(-2/3)

def v_r_orb(m_1,q,e,logP,theta,omega,i):
    """
    Orbital Radial Velocity/ Line-of-sight velocity in the center-of-mass frame of the binary orbit.

    Attributes:
        Star Parameters:
        m_1: Mass of primary star in Msun
        q  : Mass ratio of the secondary to primary, defined as m_2/m_1
        e  : Eccentricity of orbit 
        P  : Period in days
        Orbit Angles:
        θ  : True Anomaly -> phase of the orbit
        ω  : Argument of periastron/periapsis -> angle between ascending node and periapsis
        i  : Inclination

    Returns:
    v  : Radial velocity of primary star in km/s w.r.t the c.o.m.

    References:
    Implemented as in the PhD Thesis of Meghin E. Spencer (2017),
    https://deepblue.lib.umich.edu/handle/2027.42/140878.
    """
    G=(c.G.to(u.km**3/(u.Msun * u.s**2))).value
    P = (10**logP)*24*3600
    inner_power = 2*jnp.pi*G*m_1/(P*(1+q)**2)
    orientation = jnp.sin(i)*( jnp.cos(theta+omega) + e*jnp.cos(omega) )
    return q/jnp.sqrt(1-e**2) * inner_power**(1/3) * orientation

def stellar_radius(m_1, g):
    """
    m_1: Primary star mass
    g  : Surface gravity
    """
    G = None
    return jnp.sqrt(G*m_1/g)

@jax.jit
@numpyro.handlers.seed(rng_seed=key)
def binary_model_one_epoch(velocities, masses, binary_frac) -> OrbitParams:
    """ 
    Samples the orbital characteristics of binaries and returns
    the center-of-mass (COM) frame line-of-site velocity (LOSV)
    of the system for one epoch of observation. Currently using
    the distributions from DM91.

    velocities : Intrinsic velocities of the systems in the galactrocentric frame of reference in km/s.
    masses     : Mass of the primary stars in Msun.
    binary frac: Fraction of stars to assign a binary companion.
    """
    num_stars = masses.shape[0]
    # Current min/max values are for Leo II. Slightly vary depending on density of galaxay.
    min_logP = 1.57
    max_logP = 6.51

    with numpyro.plate("ps & qs", num_stars):
        q = numpyro.sample("mass ratio", dist.TruncatedNormal(loc=0.23, scale=0.42,
                                                              low=0.1, high=1.0))
        # q = numpyro.sample('mass ratio', dist.Uniform(low=0.1, high=1.0))

        logP = numpyro.sample("logP", dist.TruncatedNormal(loc=4.8, scale=2.3,
                                                           low=min_logP, high=max_logP))

    with numpyro.plate('eccentricity', num_stars):
        eccentricity = numpyro.sample('e', dist.TruncatedNormal(loc=0.31, scale=0.17))
        # eccentricity = jnp.where(
        #     logP <= 1.08, numpyro.sample("eccentricity_low", dist.Delta(0),),
        #     jnp.where(logP < 3, numpyro.sample("eccentricity_mid", dist.TruncatedNormal(loc=.25, scale=.12, 
        #                                                                                    low=0, high=max_ecc(logP)),),
        #             numpyro.sample("eccentricity_high", dist.DoublyTruncatedPowerLaw(1.0,0,max_ecc(logP)))
        #     )
        # )                                                   

    M = numpyro.sample('mean anomaly', dist.Uniform(low=0, high=2*jnp.pi),
                                                    sample_shape=masses.shape)
    true_anomaly = get_true_anomaly(M, eccentricity)
    inclination = inclination_dist(key=key, shape=masses.shape)
    periastron = numpyro.sample('periastron', dist.Uniform(low=0, high=2*jnp.pi),
                                                           sample_shape=masses.shape)

    v = v_r_orb(masses, q, eccentricity, logP, true_anomaly, periastron, inclination)
    params = OrbitParams(masses, q, eccentricity, 10**logP,
                         inclination, periastron, v, M, 
                         true_anomaly)
    return params
    # return OrbitState(OrbitParams(masses,q,eccentricity,10**logP,inclination,periastron),v,M,true_anomaly)

# masses = jnp.ones(shape=(1000,))*0.8
# state = binary_model_one_epoch(None,masses,0.5)

# vs = [[] for _ in range(len(state.v))]

# import matplotlib.pyplot as plt
# from tqdm import tqdm

# for _ in tqdm(range(100)):
#     state = state.update(40)
#     for i in range(len(state.v)):
#         vs[i].append(state.v[i])

# for _ in range(1000):
#     plt.plot(range(100), vs[_])
#     plt.show()
# plt.hist(res.params.P, bins=50)
# plt.semilogy()
# print(jnp.argmax(state.params.e))