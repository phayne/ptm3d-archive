"""Ojakangas & Stevenson (1989) Ice Shell Thermal Equilibrium Model.

Computes equilibrium ice shell thickness for Europa as a function of
colatitude and longitude, following O&S (1989, Icarus 81, 220-241).

Quick start::

    from ojakangas_stevenson import IceShellModel, MaxwellRheology

    model = IceShellModel(MaxwellRheology())
    b = model.thickness(theta=np.pi/2, lam=0)  # equatorial thickness [m]

    # Full grid
    theta, lam, b_grid = model.thickness_grid()
    print(f"Average thickness: {model.average_thickness()/1000:.1f} km")

    # With PTM3D temperatures
    from ojakangas_stevenson import PTM3DSurfaceTemperature
    ptm_temp = PTM3DSurfaceTemperature.from_binary_files('./output/temperature')
    model_ptm = IceShellModel(MaxwellRheology(), surface_temp_model=ptm_temp)

Validation::

    python -m ojakangas_stevenson.validate --plots
"""

from .shell_thickness import IceShellModel
from .rheology import MaxwellRheology, FlowLawRheology
from .surface_temperature import OSSurfaceTemperature
from .ptm3d_coupling import PTM3DSurfaceTemperature

__all__ = [
    'IceShellModel',
    'MaxwellRheology',
    'FlowLawRheology',
    'OSSurfaceTemperature',
    'PTM3DSurfaceTemperature',
]
