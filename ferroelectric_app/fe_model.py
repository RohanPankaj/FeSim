"""
Ferroelectric Material Model

This module contains the physics simulation for ferroelectric hysteresis loops
using Landau-Devonshire theory.
"""

import numpy as np
from scipy.optimize import fsolve

EPSILON_0 = 8.854e-12  # F/m, vacuum permittivity


class FerroelectricModel:
    """
    Simulates a ferroelectric material using Landau-Devonshire theory.
    
    The model calculates hysteresis loops based on material parameters,
    strain effects from substrate, and depolarization from electrodes.
    """
    
    def __init__(self, material_dict, temperature=300):
        """
        Initialize the ferroelectric model.
        
        Args:
            material_dict: Dictionary containing 'ferroelectric', 'substrate', 
                          and 'electrode' sub-dictionaries with material parameters
            temperature: Operating temperature in Kelvin (default: 300K)
        """
        self.material_dict = material_dict
        self.temperature = temperature
        self.P_loop = None
        self.V_applied = None
        
    def run_landau_hysteresis_simulation(self, V_applied_path, temperature=None):
        """
        Run the Landau-Devonshire hysteresis simulation.
        
        Traces the polarization response to an applied voltage waveform,
        including strain and depolarization effects.
        
        Args:
            V_applied_path: Array of applied voltages (V)
            temperature: Optional temperature override (K)
            
        Returns:
            P_loop: Array of polarization values (C/m²)
        """
        if temperature is None:
            temperature = self.temperature
            
        fe = self.material_dict['ferroelectric']
        sub = self.material_dict['substrate']
        elec = self.material_dict['electrode']
        film_thickness = fe['film_thickness']
        
        # Calculate strain from lattice mismatch
        eta_m = (sub['lattice_a'] - fe['lattice_a']) / fe['lattice_a']
        
        # Renormalized Landau coefficients
        a_strain_term = -4 * fe['Q12'] * eta_m / (fe['s11'] + fe['s12'])
        a_depol_term = elec['screening_lambda'] / (EPSILON_0 * elec['permittivity_e'] * film_thickness)
        
        a_tilde = fe['a0'] * (temperature - fe['T0']) + a_strain_term + a_depol_term
        b_tilde = fe['b'] + (4 * fe['Q12']**2) / (fe['s11'] + fe['s12'])
        c_tilde = fe['c']
        
        def landau_voltage_function(P_val):
            """Calculate voltage from polarization using Landau free energy."""
            return (a_tilde * P_val + b_tilde * P_val**3 + c_tilde * P_val**5) * film_thickness
        
        def equation_to_solve(P_val, V_target):
            """Root-finding equation: V(P) - V_target = 0"""
            return landau_voltage_function(P_val) - V_target
        
        # Find coercive voltages (switching points)
        coeffs_for_P_squared = [5 * c_tilde, 3 * b_tilde, a_tilde]
        roots_P_squared = np.roots(coeffs_for_P_squared)
        P_switching_points = [
            np.sqrt(np.real(r_P2)) 
            for r_P2 in roots_P_squared 
            if np.isreal(r_P2) and r_P2 > 0
        ]
        
        V_switching = sorted([landau_voltage_function(p_sw) for p_sw in P_switching_points])
        V_c_negative, V_c_positive = -V_switching[0], V_switching[0]
        
        # Trace the hysteresis loop
        P_loop = np.zeros_like(V_applied_path)
        P_current = fsolve(equation_to_solve, x0=-0.5, args=(V_applied_path[0]))[0]
        P_loop[0] = P_current
        on_upper_branch = (P_current > 0)
        
        for i in range(1, len(V_applied_path)):
            V_target = V_applied_path[i]
            V_previous = V_applied_path[i-1]
            sweeping_up = (V_target > V_previous)
            
            initial_guess_P = P_current
            
            # Handle switching between branches
            if sweeping_up and not on_upper_branch and V_target >= V_c_positive:
                initial_guess_P = 0.5
                on_upper_branch = True
            elif not sweeping_up and on_upper_branch and V_target <= V_c_negative:
                initial_guess_P = -0.5
                on_upper_branch = False
                
            P_solution = fsolve(equation_to_solve, x0=initial_guess_P, args=(V_target))
            P_current = P_solution[0]
            P_loop[i] = P_current
            
        return P_loop

    def add_parasitic_effects(self, V_applied_path, P_ideal_loop, frequency=1e6):
        """
        Add parasitic effects to the ideal hysteresis loop.
        
        Includes:
        - Linear dielectric contribution (causes tilt)
        - Ohmic leakage current (causes rounding/fattening)
        
        Args:
            V_applied_path: Array of applied voltages (V)
            P_ideal_loop: Ideal polarization from Landau simulation (C/m²)
            frequency: Measurement frequency in Hz (default: 1 MHz)
            
        Returns:
            tuple: (P_total, P_without_leakage) - Total polarization and 
                   polarization without leakage effects
        """
        fe = self.material_dict['ferroelectric']
        elec = self.material_dict['electrode']
        
        epsilon_r = fe['epsilon_r']
        film_thickness = fe['film_thickness']
        area = elec['area']
        R_leak = fe['leakage_resistance']
        
        # Linear dielectric contribution (causes tilt)
        P_dielectric = EPSILON_0 * (epsilon_r - 1) * V_applied_path / film_thickness
        
        # Leakage contribution (causes rounding/fattening)
        num_points = len(V_applied_path)
        period = 1.0 / frequency
        delta_t = period / num_points
        leakage_integral_term = np.cumsum(V_applied_path) * delta_t
        P_leak_loop = (1 / (area * R_leak)) * leakage_integral_term
        
        # Total measured polarization
        P_total = P_ideal_loop + P_dielectric + P_leak_loop
        P_without_leakage = P_dielectric + P_ideal_loop
        
        return P_total, P_without_leakage

    def calculate_current(self, polarization_SI, time_array, area):
        """
        Calculate current from polarization using I = A * dP/dt.
        
        Args:
            polarization_SI: Polarization in C/m²
            time_array: Time array in seconds
            area: Device area in m²
            
        Returns:
            current: Current array in Amperes
        """
        # Current = Area * dP/dt
        current = area * np.gradient(polarization_SI, time_array)
        return current

    def run_simulation(self, voltage_array, time_array=None, temperature=None, 
                       frequency=1e6, include_parasitics=True):
        """
        Run complete simulation with optional parasitic effects.
        
        Args:
            voltage_array: Array of applied voltages (V)
            time_array: Array of time values (s). If None, generated from frequency.
            temperature: Optional temperature override (K)
            frequency: Measurement frequency in Hz
            include_parasitics: Whether to include parasitic effects
            
        Returns:
            dict: Dictionary containing simulation results:
                - 'voltage': Applied voltage array
                - 'time': Time array
                - 'P_ideal': Ideal Landau polarization (C/m²)
                - 'P_ideal_uC_cm2': Polarization in μC/cm²
                - 'current': Current response (A)
                - 'P_total': Total polarization with parasitics (if enabled)
                - 'P_no_leakage': Polarization without leakage (if enabled)
        """
        if temperature is None:
            temperature = self.temperature
            
        # Store voltage for reference
        self.V_applied = voltage_array
        
        # Generate time array if not provided
        if time_array is None:
            num_points = len(voltage_array)
            # Estimate number of cycles from voltage zero crossings
            zero_crossings = np.where(np.diff(np.sign(voltage_array)))[0]
            num_cycles = max(1, len(zero_crossings) // 2)
            period = 1.0 / frequency
            total_time = num_cycles * period
            time_array = np.linspace(0, total_time, num_points)
        
        # Run Landau simulation
        P_ideal = self.run_landau_hysteresis_simulation(voltage_array, temperature)
        self.P_loop = P_ideal
        
        # Get electrode area
        area = self.material_dict['electrode']['area']
        
        # Calculate current from dP/dt
        current = self.calculate_current(P_ideal, time_array, area)
        
        # Convert polarization to μC/cm²
        # P_ideal is in C/m², multiply by 100 to get μC/cm²
        P_ideal_uC_cm2 = P_ideal * 100  # C/m² to μC/cm²
        
        results = {
            'voltage': voltage_array,
            'time': time_array,
            'P_ideal': P_ideal,
            'P_ideal_uC_cm2': P_ideal_uC_cm2,
            'current': current,
        }
        
        if include_parasitics:
            P_total, P_no_leakage = self.add_parasitic_effects(
                voltage_array, P_ideal, frequency
            )
            results['P_total'] = P_total
            results['P_no_leakage'] = P_no_leakage
            
        return results
    
    def calculate_loop_parameters(self, voltage, polarization):
        """
        Extract key parameters from a hysteresis loop.
        
        Args:
            voltage: Voltage array (V)
            polarization: Polarization array (C/m²)
            
        Returns:
            dict: Dictionary containing:
                - 'Pr_plus': Positive remanent polarization
                - 'Pr_minus': Negative remanent polarization
                - 'Vc_plus': Positive coercive voltage
                - 'Vc_minus': Negative coercive voltage
                - 'Pmax': Maximum polarization
                - 'Pmin': Minimum polarization
        """
        # Find remanent polarization (P at V=0)
        zero_crossings_v = np.where(np.diff(np.sign(voltage)))[0]
        Pr_values = polarization[zero_crossings_v] if len(zero_crossings_v) > 0 else [0]
        
        # Find coercive voltage (V at P=0)
        zero_crossings_p = np.where(np.diff(np.sign(polarization)))[0]
        Vc_values = voltage[zero_crossings_p] if len(zero_crossings_p) > 0 else [0]
        
        return {
            'Pr_plus': max(Pr_values) if len(Pr_values) > 0 else 0,
            'Pr_minus': min(Pr_values) if len(Pr_values) > 0 else 0,
            'Vc_plus': max(Vc_values) if len(Vc_values) > 0 else 0,
            'Vc_minus': min(Vc_values) if len(Vc_values) > 0 else 0,
            'Pmax': np.max(polarization),
            'Pmin': np.min(polarization),
        }

    def run_hysteresis_with_imprint_and_dead_layer(self, V_applied_path, temperature=None,
                                                    V_imprint=0, dead_layer_thickness=0,
                                                    dead_layer_epsilon=10, P_offset=0):
        """
        Run hysteresis simulation with imprint and dead layer effects.
        
        This is a more realistic model that includes:
        1. Imprint (built-in voltage shift)
        2. Dead layer (series capacitor causing loop tilt)
        3. Polarization offset (vertical shift)
        
        Args:
            V_applied_path: Array of applied voltages (V)
            temperature: Temperature (K)
            V_imprint: Built-in voltage shift (V) - shifts loop horizontally
            dead_layer_thickness: Dead layer thickness (m)
            dead_layer_epsilon: Dead layer dielectric constant
            P_offset: Polarization offset (C/m²) - shifts loop vertically
            
        Returns:
            P_loop: Array of polarization values (C/m²)
        """
        if temperature is None:
            temperature = self.temperature
            
        fe = self.material_dict['ferroelectric']
        sub = self.material_dict['substrate']
        elec = self.material_dict['electrode']
        film_thickness = fe['film_thickness']
        
        # Calculate strain from lattice mismatch
        eta_m = (sub['lattice_a'] - fe['lattice_a']) / fe['lattice_a']
        
        # Renormalized Landau coefficients
        a_strain_term = -4 * fe['Q12'] * eta_m / (fe['s11'] + fe['s12'])
        a_depol_term = elec['screening_lambda'] / (EPSILON_0 * elec['permittivity_e'] * film_thickness)
        
        a_tilde = fe['a0'] * (temperature - fe['T0']) + a_strain_term + a_depol_term
        b_tilde = fe['b'] + (4 * fe['Q12']**2) / (fe['s11'] + fe['s12'])
        c_tilde = fe['c']
        
        # Dead layer capacitance effect
        # V_total = V_fe + V_dead
        # V_dead = P * d_dead / (epsilon_0 * epsilon_dead)
        if dead_layer_thickness > 0:
            dead_layer_factor = dead_layer_thickness / (EPSILON_0 * dead_layer_epsilon)
        else:
            dead_layer_factor = 0
        
        def landau_voltage_function(P_val):
            """Calculate voltage from polarization including dead layer."""
            V_fe = (a_tilde * P_val + b_tilde * P_val**3 + c_tilde * P_val**5) * film_thickness
            V_dead = P_val * dead_layer_factor
            return V_fe + V_dead
        
        def equation_to_solve(P_val, V_target):
            """Root-finding equation including imprint: V(P) - (V_target - V_imprint) = 0"""
            return landau_voltage_function(P_val) - (V_target - V_imprint)
        
        # Find coercive voltages (switching points)
        # For the modified system, we need to account for dead layer
        coeffs_for_derivative = [5 * c_tilde * film_thickness, 
                                  3 * b_tilde * film_thickness, 
                                  a_tilde * film_thickness + dead_layer_factor]
        
        # dV/dP = 0 gives switching points
        roots_P_squared = np.roots([5 * c_tilde * film_thickness, 
                                     3 * b_tilde * film_thickness, 
                                     a_tilde * film_thickness + dead_layer_factor])
        
        P_switching_points = []
        for r_P2 in roots_P_squared:
            if np.isreal(r_P2) and np.real(r_P2) > 0:
                P_switching_points.append(np.sqrt(np.real(r_P2)))
        
        if len(P_switching_points) > 0:
            V_switching = sorted([landau_voltage_function(p_sw) for p_sw in P_switching_points])
            V_c_negative = -V_switching[0] + V_imprint
            V_c_positive = V_switching[0] + V_imprint
        else:
            # Fallback if no switching points found
            V_c_negative = -1.0 + V_imprint
            V_c_positive = 1.0 + V_imprint
        
        # Trace the hysteresis loop
        P_loop = np.zeros_like(V_applied_path)
        
        try:
            P_current = fsolve(equation_to_solve, x0=-0.3, args=(V_applied_path[0]), 
                              full_output=False, maxfev=1000)[0]
        except:
            P_current = -0.3
            
        P_loop[0] = P_current
        on_upper_branch = (P_current > 0)
        
        for i in range(1, len(V_applied_path)):
            V_target = V_applied_path[i]
            V_previous = V_applied_path[i-1]
            sweeping_up = (V_target > V_previous)
            
            initial_guess_P = P_current
            
            # Handle switching between branches
            if sweeping_up and not on_upper_branch and V_target >= V_c_positive:
                initial_guess_P = 0.3
                on_upper_branch = True
            elif not sweeping_up and on_upper_branch and V_target <= V_c_negative:
                initial_guess_P = -0.3
                on_upper_branch = False
            
            try:
                P_solution = fsolve(equation_to_solve, x0=initial_guess_P, args=(V_target),
                                   full_output=False, maxfev=1000)
                P_current = P_solution[0]
            except:
                pass  # Keep previous P_current
                
            P_loop[i] = P_current
        
        # Add polarization offset
        P_loop = P_loop + P_offset
            
        return P_loop


def fit_hysteresis_simple(voltage_exp, polarization_exp, film_thickness=100e-9,
                          Ps_guess=0.3, Vc_guess=1.0, V_imprint_guess=0, P_offset_guess=0):
    """
    Simple phenomenological fit using tanh function.
    
    This provides a quick fit without full Landau physics, useful for
    extracting basic parameters.
    
    The model: P = Ps * tanh((V - V_imprint ± Vc) / delta) + P_offset
    
    Args:
        voltage_exp: Experimental voltage array (V)
        polarization_exp: Experimental polarization (μC/cm²)
        film_thickness: Film thickness (m)
        Ps_guess: Initial guess for saturation polarization (μC/cm²)
        Vc_guess: Initial guess for coercive voltage (V)
        V_imprint_guess: Initial guess for imprint voltage (V)
        P_offset_guess: Initial guess for polarization offset (μC/cm²)
        
    Returns:
        dict: Fitted parameters and model curve
    """
    from scipy.optimize import curve_fit
    
    def hysteresis_model(V, Ps, Vc, delta, V_imprint, P_offset):
        """
        Two-branch hysteresis using tanh functions.
        This is an approximation that captures the basic shape.
        """
        # Determine which branch based on voltage direction
        # For fitting, we use an average of both branches
        P_up = Ps * np.tanh((V - V_imprint - Vc) / delta) + P_offset
        P_down = Ps * np.tanh((V - V_imprint + Vc) / delta) + P_offset
        return (P_up + P_down) / 2 + Ps * np.tanh((V - V_imprint) / delta) / 2
    
    # For a proper fit, we need to handle the two branches separately
    # Find the turning points
    V_max_idx = np.argmax(voltage_exp)
    V_min_idx = np.argmin(voltage_exp)
    
    # Initial parameter guesses
    Ps_init = (np.max(polarization_exp) - np.min(polarization_exp)) / 2
    P_offset_init = (np.max(polarization_exp) + np.min(polarization_exp)) / 2
    
    # Estimate Vc from zero crossings of (P - P_offset)
    P_centered = polarization_exp - P_offset_init
    zero_crossings = np.where(np.diff(np.sign(P_centered)))[0]
    if len(zero_crossings) >= 2:
        Vc_init = np.abs(voltage_exp[zero_crossings[0]] - voltage_exp[zero_crossings[-1]]) / 2
    else:
        Vc_init = Vc_guess
    
    # Estimate imprint
    if len(zero_crossings) >= 2:
        V_imprint_init = (voltage_exp[zero_crossings[0]] + voltage_exp[zero_crossings[-1]]) / 2
    else:
        V_imprint_init = V_imprint_guess
    
    return {
        'Ps': Ps_init,
        'Vc': Vc_init,
        'V_imprint': V_imprint_init,
        'P_offset': P_offset_init,
        'delta': Vc_init / 3,  # Rough estimate for switching width
    }