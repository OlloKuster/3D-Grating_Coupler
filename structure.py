import tidy3d as td
import tidy3d.plugins.adjoint as tda
import jax.numpy as jnp

from config import Config as Config
from preprocess import preprocess


def simulation_structures():
    def make_waveguide():
        waveguide = td.Structure(
            geometry=td.Box(center=(Config.lx / 2,
                                    0,
                                    -Config.rho_size[2] / 2 + Config.wg_width / 2
                                    ),
                            size=(Config.wg_length, Config.wg_width, Config.wg_width)),
            medium=td.Medium(permittivity=Config.refr_index[2] ** 2)
        )
        return waveguide

    def make_substrate():
        substrate = td.Structure(
            geometry=td.Box.from_bounds(
                rmin=(-td.inf, -td.inf, -Config.lz / 2 - Config.thickness_substrate),
                rmax=(td.inf, td.inf, -Config.rho_size[2] / 2),
            ),
            medium=td.Medium(permittivity=Config.refr_index[1] ** 2)
        )
        return substrate

    return [make_substrate(), make_waveguide()]


def make_design_region(rho, unfold, beta, rmin):
    dr_center_y = Config.rho_size[1] / 2
    dr_size_y = Config.rho_size[1]

    (nx, ny, nz) = rho.shape
    dx = Config.rho_size[0] / nx
    dy = Config.rho_size[1] / ny
    dz = Config.rho_size[2] / nz

    xmin = - Config.rho_size[0] / 2.0
    ymin = 0
    zmin = - Config.rho_size[2] / 2.0

    if unfold:
        ny = 2 * ny
        dy = 2*Config.rho_size[1] / ny
        ymin = -Config.rho_size[1]
        dr_center_y = 0
        dr_size_y = 2*Config.rho_size[1]
        rho = jnp.concat([jnp.flip(rho, axis=1), jnp.copy(rho)], axis=1)

    eps_val = (Config.refr_index[2]**2 - Config.refr_index[0]**2) * preprocess(jnp.array(rho), beta, rmin) + Config.refr_index[0]**2
    data = eps_val.reshape((Config.nx, ny, Config.nz, 1))

    xs = [xmin + index_x * dx for index_x in range(nx)]
    ys = [ymin + index_y * dy for index_y in range(ny)]
    zs = [zmin + index_z * dz for index_z in range(nz)]

    coords = dict(
        x=xs,
        y=ys,
        z=zs,
        f=[Config.freq0]
    )

    eps_components = {
        f"eps_{dim}{dim}": tda.JaxDataArray(values=data, coords=coords) for dim in "xyz"
    }
    eps_dataset = tda.JaxPermittivityDataset(**eps_components)
    eps_medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)

    geometry = tda.JaxBox(
        center=(0, dr_center_y, 0),
        size=(Config.rho_size[0], dr_size_y, Config.rho_size[2]),

    )

    custom_structure = tda.JaxStructure(
        geometry=geometry,
        medium=eps_medium
    )

    return [custom_structure]
