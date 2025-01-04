import jax
import jax.numpy as jnp
import imf
from imf.imf import make_cluster


class mass_dist():
    def __init__(self, mass_low, mass_high, num_stars) -> None:
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.num_stars = num_stars 

    def get_cluster(self, rng_key):
        cluster = imf.imf.make_cluster(10_000, mmax=self.mass_high)
        cluster = cluster[cluster > self.mass_low] 
        cluster = jax.random.choice(rng_key,cluster,shape=(self.num_stars,))
        return jnp.array(cluster)