import jax.numpy as jnp
from tidy3d.plugins.adjoint.utils.filter import ConicFilter

from config import Config


def tanh_projection(x, beta, eta=0.5):
    tanhbn = jnp.tanh(beta * eta)
    num = tanhbn + jnp.tanh(beta * (x - eta))
    den = tanhbn + jnp.tanh(beta * (1 - eta))
    return num / den


def filter_projection(x, rmin):
    conic_filter = ConicFilter(radius=rmin, design_region_dl=Config.rho_size[2] / Config.nz)
    return conic_filter.evaluate(x)


def preprocess(rho, beta=1.0, rmin=0.01):
    x = tanh_projection(filter_projection(rho, rmin), beta)
    return x
