"""
Material Parameters Database

This module contains all material property dictionaries for ferroelectric,
substrate, and electrode materials used in the simulation.
"""

# =============================================================================
# FERROELECTRIC MATERIALS
# =============================================================================

FERROELECTRIC_MATERIALS = {
    'HZO': {
        'ferroelectric': {
            # Landau Coefficients (Calibrated to PhysRevApplied.20.054007)
            'a0': 4.0e6,         # J m / (C^2 K). Positive standard Curie constant.
            'b': 4.74e10,        # J m^5 / C^4. (Matches paper's beta)
            'c': 1.0e10,         # Small positive stabilizing term
            'T0': 773.0,         # Curie Temp ~500°C (Realistic for HZO phase loss)
            # Elastic Constants (Standard HfO2)
            'Q12': -0.02,        # m^4 / C^2
            's11': 4.0e-12,      # m^2 / N
            's12': -1.2e-12,     # m^2 / N
            # Device Parameters
            'lattice_a': 0.505e-9,
            'film_thickness': 10e-9,
            'epsilon_r': 25,
            'leakage_resistance': 5e100,
        },
        'substrate': {
            'lattice_a': 0.505e-9
        },
        'electrode': {
            'screening_lambda': 0.05e-9,
            'permittivity_e': 1,
            'area': (10e-6)**2
        }
    },
    'PZT': {
        'ferroelectric': {
            'a0': 2*(-5.88e7)/(400-23),
            'b': 4*(4.764e7),
            'c': 6*(2.336e8),
            'T0': 273+400,
            'Q12': -4.6e-2,
            's11': 14.1e-12,
            's12': -4.56e-12,
            'lattice_a': 0.406e-9,
            'film_thickness': 40e-9,
            'epsilon_r': 500,
            'leakage_resistance': 5e100,
        },
        'substrate': {
            'lattice_a': 0.395e-9
        },
        'electrode': {
            'screening_lambda': 0.1e-9,
            'permittivity_e': 7.0e6,
            'area': (10e-6)**2
        }
    },
    'BTO': {
        'ferroelectric': {
            # BaTiO3 Landau coefficients
            'a0': 4.124e5,       # J m / (C^2 K)
            'b': -2.097e8,       # J m^5 / C^4
            'c': 1.294e9,        # J m^9 / C^6
            'T0': 393.0,         # Curie Temp ~120°C
            'Q12': -0.034,       # m^4 / C^2
            's11': 8.3e-12,      # m^2 / N
            's12': -2.7e-12,     # m^2 / N
            'lattice_a': 0.3992e-9,
            'film_thickness': 100e-9,
            'epsilon_r': 1500,
            'leakage_resistance': 5e100,
        },
        'substrate': {
            'lattice_a': 0.3905e-9  # SrTiO3
        },
        'electrode': {
            'screening_lambda': 0.1e-9,
            'permittivity_e': 5.0e6,
            'area': (10e-6)**2
        }
    },
}

# =============================================================================
# SUBSTRATE MATERIALS
# =============================================================================

SUBSTRATES = {
    'SI': {
        'lattice_a': 0.543e-9,
        'description': 'Silicon'
    },
    'SRO': {
        'lattice_a': 0.395e-9,
        'description': 'SrRuO3'
    },
    'TiN': {
        'lattice_a': 0.424e-9,
        'description': 'Titanium Nitride'
    },
    'STO': {
        'lattice_a': 0.3905e-9,
        'description': 'SrTiO3'
    },
    'MgO': {
        'lattice_a': 0.421e-9,
        'description': 'Magnesium Oxide'
    },
}

# =============================================================================
# ELECTRODE MATERIALS
# =============================================================================

ELECTRODES = {
    'Pt': {
        'screening_lambda': 0.05e-9,
        'permittivity_e': 1,
        'area': (10e-6)**2,
        'description': 'Platinum'
    },
    'TiN': {
        'screening_lambda': 0.08e-9,
        'permittivity_e': 1,
        'area': (10e-6)**2,
        'description': 'Titanium Nitride'
    },
    'YBCO': {
        'screening_lambda': 0.1e-9,
        'permittivity_e': 7.0e6,
        'area': (10e-6)**2,
        'description': 'YBa2Cu3O7'
    },
    'SRO': {
        'screening_lambda': 0.1e-9,
        'permittivity_e': 1e5,
        'area': (10e-6)**2,
        'description': 'SrRuO3'
    },
    'W': {
        'screening_lambda': 0.04e-9,
        'permittivity_e': 1,
        'area': (10e-6)**2,
        'description': 'Tungsten'
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ferroelectric_list():
    """Return list of available ferroelectric material names."""
    return list(FERROELECTRIC_MATERIALS.keys()) + ['Other']

def get_substrate_list():
    """Return list of available substrate material names."""
    return list(SUBSTRATES.keys()) + ['Other']

def get_electrode_list():
    """Return list of available electrode material names."""
    return list(ELECTRODES.keys()) + ['Other']

def get_ferroelectric_params(fe_name):
    """
    Get the ferroelectric parameters.
    
    Args:
        fe_name: Name of the ferroelectric material (e.g., 'HZO', 'PZT')
        
    Returns:
        dict: Ferroelectric parameters
    """
    if fe_name in FERROELECTRIC_MATERIALS:
        return FERROELECTRIC_MATERIALS[fe_name]['ferroelectric'].copy()
    return None

def get_substrate_params(sub_name):
    """
    Get substrate parameters.
    
    Args:
        sub_name: Name of substrate material
        
    Returns:
        dict: Substrate parameters
    """
    if sub_name in SUBSTRATES:
        return {k: v for k, v in SUBSTRATES[sub_name].items() if k != 'description'}
    return None

def get_electrode_params(elec_name):
    """
    Get electrode parameters.
    
    Args:
        elec_name: Name of electrode material
        
    Returns:
        dict: Electrode parameters
    """
    if elec_name in ELECTRODES:
        return {k: v for k, v in ELECTRODES[elec_name].items() if k != 'description'}
    return None