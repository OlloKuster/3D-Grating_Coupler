from dataclasses import dataclass
import tidy3d as td


@dataclass
class Config:
    wavelength = 1.00
    freq0 = td.C_0 / wavelength
    fwidth = freq0 / 10
    run_time = 50 / fwidth

    fiber_tilt = 10
    spot_size = 10.4

    rho_size = (18, 9, 4.5)
    thickness_substrate = 4
    buffer = 1 * wavelength
    dpml = 0.5

    wg_width = 2
    wg_length = 10

    lx = wg_length + rho_size[0]
    ly = wg_length + 2*rho_size[1]
    lz = thickness_substrate + rho_size[2] + buffer

    src_pos = rho_size[2] / 2 + 0.1
    mon_pos_x = rho_size[0] / 2 + 2
    mon_pos_z = - rho_size[2] / 2 + wg_width / 2

    refr_index = (1, 1.444, 1.53)
    kappa = (1e-5, 1)

    dl = wavelength / 10
    nx = 300
    ny = 150
    nz = 75


