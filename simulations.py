import jax.numpy as jnp
import tidy3d as td
import tidy3d.plugins.adjoint as tda

from structure import make_design_region, simulation_structures
from config import Config
from sources_and_monitors import ConfigSource, ConfigMonitor

from tofea.fea3d import FEA3D_T


def make_sim(rho, unfold, beta=1.0, rmin=0.01):
    eps = make_design_region(rho, unfold, beta, rmin)
    structures = simulation_structures()

    em_sim_xpol = tda.JaxSimulation(
        size=(Config.lx, Config.ly, Config.lz),
        run_time=Config.run_time,
        structures=structures,
        sources=[ConfigSource.source_xpol],
        monitors=[ConfigMonitor.mode_monitor, ConfigMonitor.field_monitor_y, ConfigMonitor.field_monitor_z,
                  ConfigMonitor.eps_monitor_y, ConfigMonitor.eps_monitor_z, ConfigMonitor.field_monitor_mode],
        output_monitors=[ConfigMonitor.fom_monitor],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=Config.wavelength),
        input_structures=eps,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()
        ),
        symmetry=(0, 1, 0)
    )

    em_sim_ypol = tda.JaxSimulation(
        size=(Config.lx, Config.ly, Config.lz),
        run_time=Config.run_time,
        structures=structures,
        sources=[ConfigSource.source_ypol],
        monitors=[ConfigMonitor.mode_monitor, ConfigMonitor.field_monitor_y, ConfigMonitor.field_monitor_z,
                  ConfigMonitor.eps_monitor_y, ConfigMonitor.eps_monitor_z, ConfigMonitor.field_monitor_mode],
        output_monitors=[ConfigMonitor.fom_monitor],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=Config.wavelength),
        input_structures=eps,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()
        ),
        symmetry=(0, -1, 0)
    )

    return em_sim_xpol, em_sim_ypol


def calculate_heat(rho):
    rho_r = 1 - rho

    shape = rho.shape
    heat_sinks_matter = jnp.zeros((shape[0] + 1,
                                   shape[1] + 1,
                                   shape[2] + 1), dtype='?')

    heat_sinks_matter = heat_sinks_matter.at[:, :, 0].set(True)

    heat_sinks_void = jnp.zeros_like(heat_sinks_matter)
    heat_sinks_void = heat_sinks_void.at[0].set(True)
    heat_sinks_void = heat_sinks_void.at[-1].set(True)
    heat_sinks_void = heat_sinks_void.at[:, 0].set(True)
    heat_sinks_void = heat_sinks_void.at[:, -1].set(True)
    heat_sinks_void = heat_sinks_void.at[:, :, -1].set(True)

    kappa_matter = Config.kappa[0] + (Config.kappa[1] - Config.kappa[0]) * rho
    kappa_void = Config.kappa[0] + (Config.kappa[1] - Config.kappa[0]) * rho_r

    fem_matter = FEA3D_T(heat_sinks_matter)
    src_matter = jnp.pad(rho, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)

    fem_void = FEA3D_T(heat_sinks_void)
    src_void = jnp.pad(rho_r, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)

    return (fem_matter, kappa_matter, src_matter), (fem_void, kappa_void, src_void)

