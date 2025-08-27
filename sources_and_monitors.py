from dataclasses import dataclass
import tidy3d as td
import numpy as np

from config import Config


@dataclass
class ConfigSource:
    source_xpol = td.GaussianBeam(
        center=(0, 0, Config.src_pos),
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=Config.freq0, fwidth=Config.fwidth),
        pol_angle=0,
        angle_theta=-Config.fiber_tilt * np.pi / 180.0,
        direction="-",
        num_freqs=1,
        waist_radius=Config.spot_size / 2,
    )

    source_ypol = td.GaussianBeam(
        center=(0, 0, Config.src_pos),
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=Config.freq0, fwidth=Config.fwidth),
        pol_angle=np.pi / 2,
        angle_theta=-Config.fiber_tilt * np.pi / 180.0,
        direction="-",
        num_freqs=1,
        waist_radius=Config.spot_size / 2,
    )


@dataclass
class ConfigMonitor:
    mode_spec = td.ModeSpec(num_modes=1, target_neff=Config.refr_index[2])
    wavelengths = np.linspace(1.5, 1.6, 21)
    freqs = td.C_0 / wavelengths
    fom_monitor = td.ModeMonitor(
        center=[Config.mon_pos_x, 0, Config.mon_pos_z],
        size=[0, 3 * Config.wg_width, 3 * Config.wg_width],
        freqs=Config.freq0,
        mode_spec=mode_spec,
        name="fom_monitor",
        colocate=False
    )

    mode_monitor = td.ModeMonitor(
        center=[Config.mon_pos_x, 0, Config.mon_pos_z],
        size=[0, 3 * Config.wg_width, 3 * Config.wg_width],
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_monitor",
    )

    field_monitor_y = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        freqs=[Config.freq0],
        name="FieldMonitor_y"
    )

    field_monitor_z = td.FieldMonitor(
        center=(0, 0, -Config.rho_size[2] / 2 + Config.wg_width / 2),
        size=(td.inf, td.inf, 0),
        freqs=[Config.freq0],
        name="FieldMonitor_z"
    )

    field_monitor_mode = td.FieldMonitor(
        center=[Config.mon_pos_x, 0, Config.mon_pos_z],
        size=[0, 3 * Config.wg_width, 3 * Config.wg_width],
        freqs=[Config.freq0],
        name="FieldMonitor_mode"
    )

    eps_monitor_y = td.PermittivityMonitor(
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        freqs=[Config.freq0],
        name="PermittivityMonitor_y"
    )

    eps_monitor_z = td.PermittivityMonitor(
        center=(0, 0, -Config.rho_size[2] / 2 + Config.wg_width / 2),
        size=(td.inf, td.inf, 0),
        freqs=[Config.freq0],
        name="PermittivityMonitor_z"
    )
