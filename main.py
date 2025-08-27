import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from objective import objective_fn, measure_temp
from optimization import optimizer_fn
from postprocess import postprocess
from simulations import calculate_heat


def main(seed):
    np.random.seed(seed)
    betas = [100, 1000, 10000, 10001]
    rmin = 0.4

    # PARAMS0 = jnp.array(np.random.rand(Config.nx, Config.ny, Config.nz))
    PARAMS0 = 0.5*jnp.ones((Config.nx, Config.ny, Config.nz))
    matter_sim, void_sim = calculate_heat(PARAMS0)
    T_matter = measure_temp(matter_sim)
    T_void = measure_temp(void_sim)

    v_matter = T_matter / 1.2
    v_void = T_void / 1.2

    dJ_fn = jax.value_and_grad(objective_fn)
    Js, params_final = optimizer_fn((PARAMS0, v_matter, v_void), dJ_fn, 60, False, betas=betas, rmin=rmin)
    plt.plot(Js)
    plt.savefig("data/loss_12.png")
    plt.close()
    postprocess(params_final, Js, "grating_coupler", max(betas), rmin)


if __name__ == "__main__":
    devices = jax.devices()
    with jax.default_device(jax.devices()[2]):
        seed = 420
        main(seed)
