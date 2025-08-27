from tidy3d.plugins.adjoint.web import run_local as run
import jax.numpy as jnp
from jax.debug import print as jprint

from simulations import make_sim, calculate_heat
from config import Config


def softplus(x, beta=50):
    mask = x * beta > 20
    return jnp.where(mask, x, 1 / beta * jnp.log(1 + jnp.exp(jnp.where(mask, 0, x * beta))))


def measure_transmission(sim_data):
    output_amps_xpol = sim_data[0].output_data[0].amps
    amp_xpol = output_amps_xpol.sel(direction="+", f=Config.freq0, mode_index=0)

    output_amps_ypol = sim_data[1].output_data[0].amps
    amp_ypol = output_amps_ypol.sel(direction="+", f=Config.freq0, mode_index=0)

    return (jnp.sum(jnp.abs(amp_xpol) ** 2) + jnp.sum(jnp.abs(amp_ypol) ** 2)) / 2


def measure_temp(sim):
    fem, kappa, src = sim
    T = fem.temperature(kappa, src)
    return jnp.sum(T) / T.size


def objective_fn(eps_params, softplus_params, step_num, unfold, beta=1, rmin=0.01):
    TARGET_MATTER, TARGET_VOID = softplus_params
    sim_em = make_sim(eps_params, unfold, beta, rmin)
    sim_temp = calculate_heat(eps_params)

    task_name_x = "grating_coupler_xpol"
    task_name_y = "grating_coupler_ypol"
    task_name_x += f"_step_{step_num}"
    task_name_y += f"_step_{step_num}"
    sim_data_xpol = run(sim_em[0], task_name=task_name_x, folder_name="grating_coupler_pol_12", verbose=False)
    sim_data_ypol = run(sim_em[1], task_name=task_name_y, folder_name="grating_coupler_pol_12", verbose=False)

    v_em = measure_transmission([sim_data_xpol, sim_data_ypol])
    v_matter = measure_temp(sim_temp[0])
    v_void = measure_temp(sim_temp[1])

    n_em = (1 - v_em) / 1  # Max transmission is 1
    n_matter = (v_matter - TARGET_MATTER) / TARGET_MATTER
    n_void = (v_void - TARGET_VOID) / TARGET_VOID

    logs = {
        "n_lens": n_em, "n_heat_m": n_matter, "n_heat_v": n_void
    }

    print("====================")
    for name, log in logs.items():
        jprint("{name}:", name=name)
        jprint("    {log}", log=log)
    print("====================")

    objs = jnp.array([n_em, n_matter, n_void])
    return jnp.linalg.norm(softplus(objs))
