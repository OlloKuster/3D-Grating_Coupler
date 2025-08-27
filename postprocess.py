import tidy3d.plugins.adjoint as tda
import h5py
import numpy as np

from config import Config
from simulations import make_sim


def postprocess(params, loss, name, beta, rmin):
    sim_opt = make_sim(params, True, beta, rmin)
    eps = sim_opt[0].epsilon(tda.JaxBox(
        center=(0, 0, 0),
        size=(Config.lx, Config.ly, Config.lz),

    ))

    with h5py.File(f"data/data_18_{name}.h5", 'w') as f:
        grp = f.create_group("grating coupler")
        grp.create_dataset("eps", data=np.real(eps))
        grp.create_dataset("loss", data=loss)
    f.close()

    with h5py.File(f"data/data_18_{name}_params.h5", 'w') as f:
        grp = f.create_group("grating coupler")
        grp.create_dataset("eps", data=np.real(params))
    f.close()
