"""
Ferroelectric Hysteresis Simulator - Main GUI Application

A graphical interface for simulating and analyzing ferroelectric hysteresis loops
using Landau-Devonshire theory.
"""

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit, minimize
from tkinter import filedialog, messagebox
import os

from fe_model import FerroelectricModel
from material_parameters import (
    FERROELECTRIC_MATERIALS,
    SUBSTRATES, 
    ELECTRODES,
    get_ferroelectric_list,
    get_substrate_list,
    get_electrode_list,
    get_ferroelectric_params,
    get_substrate_params,
    get_electrode_params,
)


class FerroelectricSimulator:
    """Main application class for the Ferroelectric Hysteresis Simulator."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Ferroelectric Hysteresis Simulator")
        self.root.geometry("1400x900")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Simulation tab variables
        self.param_entries = {}
        self.simulation_results = None
        self.simulation_df = None
        
        # Analysis tab variables
        self.analysis_param_entries = {}
        self.experimental_data = None
        self.analysis_results = None
        self.fitted_model = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main user interface."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab view at the top
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("Simulation")
        self.tabview.add("Analysis")
        
        # Setup both tabs
        self.setup_simulation_tab()
        self.setup_analysis_tab()
        
    # =========================================================================
    # SIMULATION TAB
    # =========================================================================
    
    def setup_simulation_tab(self):
        """Set up the Simulation tab UI."""
        sim_tab = self.tabview.tab("Simulation")
        
        # Top control panel
        control_frame = ctk.CTkFrame(sim_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # FE Material dropdown
        ctk.CTkLabel(control_frame, text="FE Material:").grid(row=0, column=0, padx=5, pady=5)
        self.fe_material = ctk.CTkComboBox(
            control_frame, 
            values=get_ferroelectric_list(),
            width=150,
            command=self.update_ferroelectric_parameters
        )
        self.fe_material.grid(row=0, column=1, padx=5, pady=5)
        self.fe_material.set("HZO")
        
        # Substrate dropdown
        ctk.CTkLabel(control_frame, text="Substrate:").grid(row=0, column=2, padx=5, pady=5)
        self.substrate = ctk.CTkComboBox(
            control_frame,
            values=get_substrate_list(),
            width=150,
            command=self.update_substrate_parameters
        )
        self.substrate.grid(row=0, column=3, padx=5, pady=5)
        self.substrate.set("SI")
        
        # Electrode dropdown
        ctk.CTkLabel(control_frame, text="Electrode:").grid(row=0, column=4, padx=5, pady=5)
        self.electrode = ctk.CTkComboBox(
            control_frame,
            values=get_electrode_list(),
            width=150,
            command=self.update_electrode_parameters
        )
        self.electrode.grid(row=0, column=5, padx=5, pady=5)
        self.electrode.set("Pt")
        
        # Run Simulation button
        self.run_btn = ctk.CTkButton(
            control_frame,
            text="Run Simulation",
            command=self.run_simulation
        )
        self.run_btn.grid(row=0, column=6, padx=20, pady=5)
        
        # Save Data button
        self.save_btn = ctk.CTkButton(
            control_frame,
            text="Save Data",
            command=self.save_data,
            state="disabled"
        )
        self.save_btn.grid(row=0, column=7, padx=10, pady=5)
        
        # Save Plots button
        self.save_plots_btn = ctk.CTkButton(
            control_frame,
            text="Save Plots",
            command=self.save_plots,
            state="disabled"
        )
        self.save_plots_btn.grid(row=0, column=8, padx=10, pady=5)
        
        # Main content area
        content_frame = ctk.CTkFrame(sim_tab)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Graph area
        graph_frame = ctk.CTkFrame(content_frame)
        graph_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Graph type dropdown
        graph_control = ctk.CTkFrame(graph_frame)
        graph_control.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(graph_control, text="Graph Type:").pack(side="left", padx=5)
        self.graph_type = ctk.CTkComboBox(
            graph_control,
            values=["P-E Loop", "P-V Loop", "Applied Voltage", "Current vs Time", "Current vs Voltage"],
            command=self.update_graph,
            width=200
        )
        self.graph_type.pack(side="left", padx=5)
        self.graph_type.set("P-E Loop")
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right side panel
        side_panel = ctk.CTkFrame(content_frame, width=350)
        side_panel.pack(side="right", fill="both")
        side_panel.pack_propagate(False)
        
        # Parameters section header
        params_label = ctk.CTkLabel(side_panel, text="Simulation Parameters", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        params_label.pack(pady=10)
        
        # Scrollable frame for parameters
        scroll_frame = ctk.CTkScrollableFrame(side_panel, height=500)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Waveform parameters
        wave_label = ctk.CTkLabel(scroll_frame, text="Waveform Parameters", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        wave_label.pack(pady=(5, 5))
        
        wave_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        wave_frame.pack(fill="x", padx=5, pady=3)
        ctk.CTkLabel(wave_frame, text="Waveform Type:", width=150, anchor="w").pack(side="left")
        self.waveform_type = ctk.CTkComboBox(wave_frame, values=["Triangle", "Sine"], width=150)
        self.waveform_type.pack(side="right")
        self.waveform_type.set("Triangle")
        
        self.create_parameter_input(scroll_frame, "Amplitude (V):", "amplitude", "3")
        self.create_parameter_input(scroll_frame, "Frequency (Hz):", "frequency", "1000")
        self.create_parameter_input(scroll_frame, "Cycles:", "cycles", "2")
        self.create_parameter_input(scroll_frame, "Temperature (K):", "temperature", "300")
        
        # Separator
        separator = ctk.CTkFrame(scroll_frame, height=2, fg_color="gray40")
        separator.pack(fill="x", padx=5, pady=15)
        
        # Device parameters
        fe_label = ctk.CTkLabel(scroll_frame, text="Ferroelectric Properties", 
                               font=ctk.CTkFont(size=14, weight="bold"))
        fe_label.pack(pady=(5, 5))
        
        self.create_parameter_input(scroll_frame, "a0 (J·m/C²·K):", "a0", "")
        self.create_parameter_input(scroll_frame, "b (J·m⁵/C⁴):", "b", "")
        self.create_parameter_input(scroll_frame, "c (J·m⁹/C⁶):", "c", "")
        self.create_parameter_input(scroll_frame, "T0 (K):", "T0", "")
        self.create_parameter_input(scroll_frame, "Q12 (m⁴/C²):", "Q12", "")
        self.create_parameter_input(scroll_frame, "s11 (m²/N):", "s11", "")
        self.create_parameter_input(scroll_frame, "s12 (m²/N):", "s12", "")
        self.create_parameter_input(scroll_frame, "Lattice a (m):", "fe_lattice_a", "")
        self.create_parameter_input(scroll_frame, "Film Thickness (m):", "film_thickness", "")
        self.create_parameter_input(scroll_frame, "Background Permittivityε_r:", "epsilon_r", "")
        self.create_parameter_input(scroll_frame, "Leakage R (Ω):", "leakage_resistance", "")
        
        # Substrate parameters
        sub_label = ctk.CTkLabel(scroll_frame, text="Substrate Properties", 
                                font=ctk.CTkFont(size=14, weight="bold"))
        sub_label.pack(pady=(15, 5))
        self.create_parameter_input(scroll_frame, "Lattice a (m):", "sub_lattice_a", "")
        
        # Electrode parameters
        elec_label = ctk.CTkLabel(scroll_frame, text="Electrode Properties", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        elec_label.pack(pady=(15, 5))
        self.create_parameter_input(scroll_frame, "Screening λ (m):", "screening_lambda", "")
        self.create_parameter_input(scroll_frame, "Electron Permittivity:", "permittivity_e", "")
        self.create_parameter_input(scroll_frame, "Area (m²):", "area", "")
        
        # Results section
        results_label = ctk.CTkLabel(side_panel, text="Results", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        results_label.pack(pady=(10, 5))
        
        self.results_text = ctk.CTkTextbox(side_panel, height=150)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize with default parameters
        self.update_all_parameters()
        self.update_graph()

    # =========================================================================
    # ANALYSIS TAB
    # =========================================================================
    
    def setup_analysis_tab(self):
        """Set up the Analysis tab UI."""
        analysis_tab = self.tabview.tab("Analysis")
        
        # Top control panel
        control_frame = ctk.CTkFrame(analysis_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Load Data button
        self.load_data_btn = ctk.CTkButton(
            control_frame,
            text="Load Experimental Data",
            command=self.load_experimental_data
        )
        self.load_data_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # Data file label
        self.data_file_label = ctk.CTkLabel(control_frame, text="No data loaded")
        self.data_file_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Run Analysis button
        self.run_analysis_btn = ctk.CTkButton(
            control_frame,
            text="Run Analysis",
            command=self.run_analysis,
            state="disabled"
        )
        self.run_analysis_btn.grid(row=0, column=2, padx=20, pady=5)
        
        # Fit Model button
        self.fit_model_btn = ctk.CTkButton(
            control_frame,
            text="Fit Landau Model",
            command=self.fit_landau_model,
            state="disabled"
        )
        self.fit_model_btn.grid(row=0, column=3, padx=10, pady=5)
        
        # Save Analysis button
        self.save_analysis_btn = ctk.CTkButton(
            control_frame,
            text="Save Analysis",
            command=self.save_analysis,
            state="disabled"
        )
        self.save_analysis_btn.grid(row=0, column=4, padx=10, pady=5)
        
        # Material selection row
        material_frame = ctk.CTkFrame(analysis_tab)
        material_frame.pack(fill="x", padx=10, pady=5)
        
        # FE Material dropdown
        ctk.CTkLabel(material_frame, text="FE Material:").grid(row=0, column=0, padx=5, pady=5)
        self.analysis_fe_material = ctk.CTkComboBox(
            material_frame, 
            values=get_ferroelectric_list(),
            width=120,
            command=self.update_analysis_fe_parameters
        )
        self.analysis_fe_material.grid(row=0, column=1, padx=5, pady=5)
        self.analysis_fe_material.set("PZT")
        
        # Substrate dropdown
        ctk.CTkLabel(material_frame, text="Substrate:").grid(row=0, column=2, padx=5, pady=5)
        self.analysis_substrate = ctk.CTkComboBox(
            material_frame,
            values=get_substrate_list(),
            width=120,
            command=self.update_analysis_substrate_parameters
        )
        self.analysis_substrate.grid(row=0, column=3, padx=5, pady=5)
        self.analysis_substrate.set("SRO")
        
        # Electrode dropdown
        ctk.CTkLabel(material_frame, text="Electrode:").grid(row=0, column=4, padx=5, pady=5)
        self.analysis_electrode = ctk.CTkComboBox(
            material_frame,
            values=get_electrode_list(),
            width=120,
            command=self.update_analysis_electrode_parameters
        )
        self.analysis_electrode.grid(row=0, column=5, padx=5, pady=5)
        self.analysis_electrode.set("YBCO")
        
        # Main content area
        content_frame = ctk.CTkFrame(analysis_tab)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Graph area
        graph_frame = ctk.CTkFrame(content_frame)
        graph_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Graph type dropdown
        graph_control = ctk.CTkFrame(graph_frame)
        graph_control.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(graph_control, text="Graph Type:").pack(side="left", padx=5)
        self.analysis_graph_type = ctk.CTkComboBox(
            graph_control,
            values=["P-V Loop", "P-E Loop", "Current vs Time", "Current vs Voltage", 
                    "Leakage Corrected P-V", "Model Fit Comparison"],
            command=self.update_analysis_graph,
            width=200
        )
        self.analysis_graph_type.pack(side="left", padx=5)
        self.analysis_graph_type.set("P-V Loop")
        
        # Matplotlib figure for analysis
        self.analysis_figure = Figure(figsize=(8, 6), dpi=100)
        self.analysis_ax = self.analysis_figure.add_subplot(111)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_figure, graph_frame)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right side panel
        side_panel = ctk.CTkFrame(content_frame, width=380)
        side_panel.pack(side="right", fill="both")
        side_panel.pack_propagate(False)
        
        # Analysis Parameters header
        params_label = ctk.CTkLabel(side_panel, text="Analysis Parameters", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        params_label.pack(pady=10)
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(side_panel, height=300)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Device parameters for fitting
        device_label = ctk.CTkLabel(scroll_frame, text="Device Parameters", 
                                   font=ctk.CTkFont(size=14, weight="bold"))
        device_label.pack(pady=(5, 5))
        
        self.create_analysis_parameter_input(scroll_frame, "Film Thickness (nm):", "a_film_thickness", "100")
        self.create_analysis_parameter_input(scroll_frame, "Device Area (μm²):", "a_area", "400")
        self.create_analysis_parameter_input(scroll_frame, "Temperature (K):", "a_temperature", "300")
        self.create_analysis_parameter_input(scroll_frame, "ε_r (dielectric):", "a_epsilon_r", "500")
        
        # Leakage Analysis
        leakage_label = ctk.CTkLabel(scroll_frame, text="Leakage Analysis", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        leakage_label.pack(pady=(15, 5))
        
        self.create_analysis_parameter_input(scroll_frame, "Leakage R (Ω):", "a_leakage_r", "1e12")
        
        # Enable leakage correction checkbox
        self.leakage_correction_var = ctk.BooleanVar(value=False)
        leakage_check = ctk.CTkCheckBox(scroll_frame, text="Enable Leakage Correction",
                                        variable=self.leakage_correction_var)
        leakage_check.pack(pady=5)
        
        # Dead Layer Analysis
        dead_label = ctk.CTkLabel(scroll_frame, text="Dead Layer Analysis", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        dead_label.pack(pady=(15, 5))
        
        self.create_analysis_parameter_input(scroll_frame, "Dead Layer (nm):", "a_dead_layer", "1.0")
        self.create_analysis_parameter_input(scroll_frame, "Dead Layer ε_r:", "a_dead_epsilon", "10")
        
        # Imprint Analysis
        imprint_label = ctk.CTkLabel(scroll_frame, text="Imprint Analysis", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        imprint_label.pack(pady=(15, 5))
        
        self.create_analysis_parameter_input(scroll_frame, "V_imprint (V):", "a_v_imprint", "0")
        self.create_analysis_parameter_input(scroll_frame, "P_offset (μC/cm²):", "a_p_offset", "0")
        
        # Auto-estimate button
        auto_estimate_btn = ctk.CTkButton(
            scroll_frame,
            text="Auto-Estimate from Data",
            command=self.auto_estimate_imprint,
            width=200
        )
        auto_estimate_btn.pack(pady=10)
        
        # Model Fitting Parameters (Landau coefficients)
        fit_label = ctk.CTkLabel(scroll_frame, text="Landau Coefficients", 
                                font=ctk.CTkFont(size=14, weight="bold"))
        fit_label.pack(pady=(15, 5))
        
        self.create_analysis_parameter_input(scroll_frame, "a₀ (J·m/C²·K):", "a_fit_a0", "-3.12e5")
        self.create_analysis_parameter_input(scroll_frame, "b (J·m⁵/C⁴):", "a_fit_b", "1.91e8")
        self.create_analysis_parameter_input(scroll_frame, "c (J·m⁹/C⁶):", "a_fit_c", "1.40e9")
        self.create_analysis_parameter_input(scroll_frame, "T₀ (K):", "a_fit_T0", "673")
        
        # Results section
        results_label = ctk.CTkLabel(side_panel, text="Analysis Results", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        results_label.pack(pady=(10, 5))
        
        self.analysis_results_text = ctk.CTkTextbox(side_panel, height=200)
        self.analysis_results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize analysis parameters from selected materials
        self.update_analysis_fe_parameters()
        self.update_analysis_substrate_parameters()
        self.update_analysis_electrode_parameters()
        
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
        
    def create_parameter_input(self, parent, label_text, key, default_value):
        """Create a labeled parameter input field for simulation tab."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=3)
        
        label = ctk.CTkLabel(frame, text=label_text, width=150, anchor="w")
        label.pack(side="left")
        
        entry = ctk.CTkEntry(frame, width=150)
        entry.insert(0, default_value)
        entry.pack(side="right")
        
        self.param_entries[key] = entry
        
    def create_analysis_parameter_input(self, parent, label_text, key, default_value):
        """Create a labeled parameter input field for analysis tab."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=3)
        
        label = ctk.CTkLabel(frame, text=label_text, width=150, anchor="w")
        label.pack(side="left")
        
        entry = ctk.CTkEntry(frame, width=150)
        entry.insert(0, default_value)
        entry.pack(side="right")
        
        self.analysis_param_entries[key] = entry
        
    def clear_ferroelectric_params(self):
        """Clear all ferroelectric parameter fields."""
        fe_params = ['a0', 'b', 'c', 'T0', 'Q12', 's11', 's12', 
                     'fe_lattice_a', 'film_thickness', 'epsilon_r', 'leakage_resistance']
        for param in fe_params:
            self.param_entries[param].delete(0, 'end')
            
    def clear_substrate_params(self):
        """Clear all substrate parameter fields."""
        self.param_entries['sub_lattice_a'].delete(0, 'end')
        
    def clear_electrode_params(self):
        """Clear all electrode parameter fields."""
        elec_params = ['screening_lambda', 'permittivity_e', 'area']
        for param in elec_params:
            self.param_entries[param].delete(0, 'end')

    def update_ferroelectric_parameters(self, *args):
        """Update only ferroelectric parameter fields."""
        fe_mat = self.fe_material.get()
        
        if fe_mat == "Other":
            self.clear_ferroelectric_params()
        else:
            fe_params = get_ferroelectric_params(fe_mat)
            if fe_params:
                self.param_entries['a0'].delete(0, 'end')
                self.param_entries['a0'].insert(0, f"{fe_params['a0']:.6e}")
                self.param_entries['b'].delete(0, 'end')
                self.param_entries['b'].insert(0, f"{fe_params['b']:.6e}")
                self.param_entries['c'].delete(0, 'end')
                self.param_entries['c'].insert(0, f"{fe_params['c']:.6e}")
                self.param_entries['T0'].delete(0, 'end')
                self.param_entries['T0'].insert(0, f"{fe_params['T0']:.2f}")
                self.param_entries['Q12'].delete(0, 'end')
                self.param_entries['Q12'].insert(0, f"{fe_params['Q12']:.6e}")
                self.param_entries['s11'].delete(0, 'end')
                self.param_entries['s11'].insert(0, f"{fe_params['s11']:.6e}")
                self.param_entries['s12'].delete(0, 'end')
                self.param_entries['s12'].insert(0, f"{fe_params['s12']:.6e}")
                self.param_entries['fe_lattice_a'].delete(0, 'end')
                self.param_entries['fe_lattice_a'].insert(0, f"{fe_params['lattice_a']:.6e}")
                self.param_entries['film_thickness'].delete(0, 'end')
                self.param_entries['film_thickness'].insert(0, f"{fe_params['film_thickness']:.6e}")
                self.param_entries['epsilon_r'].delete(0, 'end')
                self.param_entries['epsilon_r'].insert(0, f"{fe_params['epsilon_r']:.2f}")
                self.param_entries['leakage_resistance'].delete(0, 'end')
                self.param_entries['leakage_resistance'].insert(0, f"{fe_params['leakage_resistance']:.2e}")
    
    def update_substrate_parameters(self, *args):
        """Update only substrate parameter fields."""
        sub_mat = self.substrate.get()
        
        if sub_mat == "Other":
            self.clear_substrate_params()
        else:
            sub_params = get_substrate_params(sub_mat)
            if sub_params:
                self.param_entries['sub_lattice_a'].delete(0, 'end')
                self.param_entries['sub_lattice_a'].insert(0, f"{sub_params['lattice_a']:.6e}")
    
    def update_electrode_parameters(self, *args):
        """Update only electrode parameter fields."""
        elec_mat = self.electrode.get()
        
        if elec_mat == "Other":
            self.clear_electrode_params()
        else:
            elec_params = get_electrode_params(elec_mat)
            if elec_params:
                self.param_entries['screening_lambda'].delete(0, 'end')
                self.param_entries['screening_lambda'].insert(0, f"{elec_params['screening_lambda']:.6e}")
                self.param_entries['permittivity_e'].delete(0, 'end')
                self.param_entries['permittivity_e'].insert(0, f"{elec_params['permittivity_e']:.6e}")
                self.param_entries['area'].delete(0, 'end')
                self.param_entries['area'].insert(0, f"{elec_params['area']:.6e}")
                
    def update_all_parameters(self):
        """Update all parameter fields based on selected materials."""
        self.update_ferroelectric_parameters()
        self.update_substrate_parameters()
        self.update_electrode_parameters()
        
    def update_analysis_fe_parameters(self, *args):
        """Update analysis parameters when FE material changes."""
        fe_mat = self.analysis_fe_material.get()
        
        if fe_mat != "Other":
            fe_params = get_ferroelectric_params(fe_mat)
            if fe_params:
                # Update Landau coefficients
                self.analysis_param_entries['a_fit_a0'].delete(0, 'end')
                self.analysis_param_entries['a_fit_a0'].insert(0, f"{fe_params['a0']:.2e}")
                self.analysis_param_entries['a_fit_b'].delete(0, 'end')
                self.analysis_param_entries['a_fit_b'].insert(0, f"{fe_params['b']:.2e}")
                self.analysis_param_entries['a_fit_c'].delete(0, 'end')
                self.analysis_param_entries['a_fit_c'].insert(0, f"{fe_params['c']:.2e}")
                self.analysis_param_entries['a_fit_T0'].delete(0, 'end')
                self.analysis_param_entries['a_fit_T0'].insert(0, f"{fe_params['T0']:.1f}")
                # Update film thickness (convert to nm for display)
                self.analysis_param_entries['a_film_thickness'].delete(0, 'end')
                self.analysis_param_entries['a_film_thickness'].insert(0, f"{fe_params['film_thickness']*1e9:.1f}")
                # Update dielectric constant
                self.analysis_param_entries['a_epsilon_r'].delete(0, 'end')
                self.analysis_param_entries['a_epsilon_r'].insert(0, f"{fe_params['epsilon_r']:.1f}")
                
    def update_analysis_substrate_parameters(self, *args):
        """Update analysis parameters when substrate changes."""
        # Currently just for display/reference - could add lattice mismatch calculations
        pass
        
    def update_analysis_electrode_parameters(self, *args):
        """Update analysis parameters when electrode changes."""
        # Could add screening length effects here
        pass

    # =========================================================================
    # SIMULATION TAB METHODS
    # =========================================================================
        
    def update_graph(self, *args):
        """Update the simulation graph."""
        self.ax.clear()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        graph_type = self.graph_type.get()
        
        if self.simulation_results is not None:
            voltage = self.simulation_results['voltage']
            time = self.simulation_results['time']
            P_uC_cm2 = self.simulation_results['P_ideal_uC_cm2']
            current = self.simulation_results['current']
            film_thickness = float(self.param_entries['film_thickness'].get())
            
            if graph_type == "P-E Loop":
                E_field = (voltage / film_thickness) / 1e8
                self.ax.plot(E_field, P_uC_cm2, 'b-', linewidth=2)
                self.ax.set_xlabel('Electric Field (MV/cm)', fontsize=36)
                self.ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.ax.set_title('P-E Hysteresis Loop (Simulated)', fontsize=40)
                
            elif graph_type == "P-V Loop":
                self.ax.plot(voltage, P_uC_cm2, 'b-', linewidth=2)
                self.ax.set_xlabel('Voltage (V)', fontsize=36)
                self.ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.ax.set_title('P-V Hysteresis Loop (Simulated)', fontsize=40)
                
            elif graph_type == "Applied Voltage":
                time_plot = time * 1000
                self.ax.plot(time_plot, voltage, 'g-', linewidth=2)
                self.ax.set_xlabel('Time (ms)', fontsize=36)
                self.ax.set_ylabel('Voltage (V)', fontsize=36)
                self.ax.set_title('Applied Voltage vs Time', fontsize=40)
                
            elif graph_type == "Current vs Time":
                time_plot = time * 1000
                current_uA = current * 1e6
                self.ax.plot(time_plot, current_uA, 'r-', linewidth=2)
                self.ax.set_xlabel('Time (ms)', fontsize=36)
                self.ax.set_ylabel('Current (μA)', fontsize=36)
                self.ax.set_title('Current vs Time (Simulated)', fontsize=40)
                
            elif graph_type == "Current vs Voltage":
                current_uA = current * 1e6
                self.ax.plot(voltage, current_uA, 'r-', linewidth=2)
                self.ax.set_xlabel('Voltage (V)', fontsize=36)
                self.ax.set_ylabel('Current (μA)', fontsize=36)
                self.ax.set_title('Current vs Voltage (Simulated)', fontsize=40)
        else:
            # Placeholder graphs
            if graph_type == "P-E Loop":
                E = np.linspace(-5, 5, 1000)
                P = 20 * np.tanh((E - 1.5) / 0.5) + 20 * np.tanh((E + 1.5) / 0.5)
                self.ax.plot(E, P, 'b-', linewidth=2, alpha=0.5)
                self.ax.set_xlabel('Electric Field (MV/cm)', fontsize=36)
                self.ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.ax.set_title('P-E Hysteresis Loop (Placeholder)', fontsize=40)
            elif graph_type == "P-V Loop":
                V = np.linspace(-3, 3, 1000)
                P = 20 * np.tanh((V - 1) / 0.3) + 20 * np.tanh((V + 1) / 0.3)
                self.ax.plot(V, P, 'b-', linewidth=2, alpha=0.5)
                self.ax.set_xlabel('Voltage (V)', fontsize=36)
                self.ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.ax.set_title('P-V Hysteresis Loop (Placeholder)', fontsize=40)
            elif graph_type == "Applied Voltage":
                t = np.linspace(0, 2, 1000)
                V = 3 * signal.sawtooth(2 * np.pi * t, width=0.5)
                self.ax.plot(t, V, 'g-', linewidth=2, alpha=0.5)
                self.ax.set_xlabel('Time (ms)', fontsize=36)
                self.ax.set_ylabel('Voltage (V)', fontsize=36)
                self.ax.set_title('Applied Voltage vs Time (Placeholder)', fontsize=40)
            elif graph_type == "Current vs Time":
                t = np.linspace(0, 2, 1000)
                I = 5 * np.sin(4 * np.pi * t) * np.exp(-t/2)
                self.ax.plot(t, I, 'r-', linewidth=2, alpha=0.5)
                self.ax.set_xlabel('Time (ms)', fontsize=36)
                self.ax.set_ylabel('Current (μA)', fontsize=36)
                self.ax.set_title('Current vs Time (Placeholder)', fontsize=40)
            elif graph_type == "Current vs Voltage":
                V = np.linspace(-3, 3, 1000)
                I = 2 * np.sin(V * 2) + 0.5 * V
                self.ax.plot(V, I, 'r-', linewidth=2, alpha=0.5)
                self.ax.set_xlabel('Voltage (V)', fontsize=36)
                self.ax.set_ylabel('Current (μA)', fontsize=36)
                self.ax.set_title('Current vs Voltage (Placeholder)', fontsize=40)
        
        self.ax.tick_params(labelsize=36)
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def build_material_dict(self):
        """Build material dictionary from current parameter entries."""
        return {
            'ferroelectric': {
                'a0': float(self.param_entries['a0'].get()),
                'b': float(self.param_entries['b'].get()),
                'c': float(self.param_entries['c'].get()),
                'T0': float(self.param_entries['T0'].get()),
                'Q12': float(self.param_entries['Q12'].get()),
                's11': float(self.param_entries['s11'].get()),
                's12': float(self.param_entries['s12'].get()),
                'lattice_a': float(self.param_entries['fe_lattice_a'].get()),
                'film_thickness': float(self.param_entries['film_thickness'].get()),
                'epsilon_r': float(self.param_entries['epsilon_r'].get()),
                'leakage_resistance': float(self.param_entries['leakage_resistance'].get()),
            },
            'substrate': {
                'lattice_a': float(self.param_entries['sub_lattice_a'].get()),
            },
            'electrode': {
                'screening_lambda': float(self.param_entries['screening_lambda'].get()),
                'permittivity_e': float(self.param_entries['permittivity_e'].get()),
                'area': float(self.param_entries['area'].get()),
            }
        }
        
    def run_simulation(self):
        """Run the ferroelectric simulation."""
        try:
            waveform = self.waveform_type.get()
            amplitude = float(self.param_entries['amplitude'].get())
            frequency = float(self.param_entries['frequency'].get())
            num_cycles = int(float(self.param_entries['cycles'].get()))
            temperature = float(self.param_entries['temperature'].get())
            
            period = 1.0 / frequency
            total_time = num_cycles * period
            samples_per_cycle = 500
            num_points = samples_per_cycle * num_cycles
            time = np.linspace(0, total_time, num_points)
            
            if waveform == "Triangle":
                voltage = amplitude * signal.sawtooth(2 * np.pi * frequency * time, width=0.5)
            else:
                voltage = amplitude * np.sin(2 * np.pi * frequency * time)
            
            material_dict = self.build_material_dict()
            model = FerroelectricModel(material_dict, temperature=temperature)
            self.simulation_results = model.run_simulation(
                voltage, time_array=time, temperature=temperature,
                frequency=frequency, include_parasitics=True
            )
            
            self.simulation_df = pd.DataFrame({
                'time (s)': time,
                'applied voltage (V)': voltage,
                'Polarization Response (uC/cm2)': self.simulation_results['P_ideal_uC_cm2'],
                'Current Response (A)': self.simulation_results['current']
            })
            
            self.save_btn.configure(state="normal")
            self.save_plots_btn.configure(state="normal")
            
            loop_params = model.calculate_loop_parameters(
                self.simulation_results['voltage'],
                self.simulation_results['P_ideal']
            )
            
            self.results_text.delete("1.0", "end")
            Pr_plus_uC = loop_params['Pr_plus'] * 100
            Pr_minus_uC = loop_params['Pr_minus'] * 100
            Pmax_uC = loop_params['Pmax'] * 100
            Pmin_uC = loop_params['Pmin'] * 100
            peak_current = np.max(np.abs(self.simulation_results['current']))
            
            results = f"""Simulation Complete!

Material: {self.fe_material.get()}
Substrate: {self.substrate.get()}
Electrode: {self.electrode.get()}

Waveform:
  Type: {waveform}
  Amplitude: {amplitude} V
  Frequency: {frequency} Hz
  Cycles: {num_cycles}
  Temperature: {temperature} K

Loop Parameters:
  Pr+: {Pr_plus_uC:.2f} μC/cm²
  Pr-: {Pr_minus_uC:.2f} μC/cm²
  Vc+: {loop_params['Vc_plus']:.3f} V
  Vc-: {loop_params['Vc_minus']:.3f} V
  Pmax: {Pmax_uC:.2f} μC/cm²
  Pmin: {Pmin_uC:.2f} μC/cm²
  Peak Current: {peak_current*1e6:.3f} μA
"""
            self.results_text.insert("1.0", results)
            self.update_graph()
            
        except Exception as e:
            messagebox.showerror(
                "Simulation Failed",
                f"The simulation failed.\n\nPlease check your parameters.\n\nError: {str(e)}"
            )
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Simulation Error:\n{str(e)}")
            
    def save_data(self):
        """Save simulation data to CSV."""
        if self.simulation_df is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Simulation Data",
            initialfile=f"ferroelectric_simulation_{self.fe_material.get()}.csv"
        )
        
        if file_path:
            try:
                self.simulation_df.to_csv(file_path, index=False)
                current_text = self.results_text.get("1.0", "end")
                self.results_text.delete("1.0", "end")
                self.results_text.insert("1.0", current_text.strip() + f"\n\nData saved to:\n{os.path.basename(file_path)}")
            except Exception as e:
                self.results_text.insert("end", f"\n\nError saving: {str(e)}")
                
    def save_plots(self):
        """Save all simulation plots."""
        if self.simulation_results is None:
            return
            
        dir_path = filedialog.askdirectory(title="Select Directory to Save Plots")
        if not dir_path:
            return
            
        try:
            voltage = self.simulation_results['voltage']
            time = self.simulation_results['time']
            P_uC_cm2 = self.simulation_results['P_ideal_uC_cm2']
            current = self.simulation_results['current']
            film_thickness = float(self.param_entries['film_thickness'].get())
            
            E_field = (voltage / film_thickness) / 1e8
            time_ms = time * 1000
            current_uA = current * 1e6
            prefix = f"{self.fe_material.get()}_{self.substrate.get()}_{self.electrode.get()}"
            
            # Save all 5 plots
            for plot_type, data, labels, fname in [
                ("P-E", (E_field, P_uC_cm2), ('Electric Field (MV/cm)', 'Polarization (μC/cm²)'), "PE_loop"),
                ("P-V", (voltage, P_uC_cm2), ('Voltage (V)', 'Polarization (μC/cm²)'), "PV_loop"),
                ("V-t", (time_ms, voltage), ('Time (ms)', 'Voltage (V)'), "voltage_vs_time"),
                ("I-t", (time_ms, current_uA), ('Time (ms)', 'Current (μA)'), "current_vs_time"),
                ("I-V", (voltage, current_uA), ('Voltage (V)', 'Current (μA)'), "current_vs_voltage"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                ax.plot(data[0], data[1], 'b-' if 'P' in plot_type else ('g-' if 'V-t' in plot_type else 'r-'), linewidth=2)
                ax.set_xlabel(labels[0], fontsize=36)
                ax.set_ylabel(labels[1], fontsize=36)
                ax.set_title(f'{plot_type} Loop', fontsize=40)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                fig.tight_layout()
                fig.savefig(os.path.join(dir_path, f"{prefix}_{fname}.png"))
                plt.close(fig)
                
            current_text = self.results_text.get("1.0", "end")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", current_text.strip() + f"\n\n5 plots saved to:\n{dir_path}")
            
        except Exception as e:
            self.results_text.insert("end", f"\n\nError saving plots: {str(e)}")

    # =========================================================================
    # ANALYSIS TAB METHODS
    # =========================================================================
    
    def load_experimental_data(self):
        """Load experimental hysteresis data from CSV."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Load Experimental Data"
        )
        
        if not file_path:
            return
            
        try:
            # Try to read the file, handling the header rows
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the row with actual column headers (look for 'time' AND 'polarization')
            header_row = None
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Check if this line contains the column headers we need
                if 'time' in line_lower and ('polarization' in line_lower or 'polar' in line_lower):
                    header_row = i
                    print(f"Found header at line {i+1} (0-indexed: {i}): {line.strip()}")
                    break
            
            # If not found with polarization, try time + voltage + current
            if header_row is None:
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    # Make sure it's not the metadata row by checking it doesn't have 'frequency' or 'n_cycles'
                    if ('time' in line_lower and 'voltage' in line_lower and 'current' in line_lower 
                        and 'frequency' not in line_lower and 'n_cycles' not in line_lower):
                        header_row = i
                        print(f"Found header at line {i+1} (0-indexed: {i}): {line.strip()}")
                        break
            
            if header_row is None:
                raise ValueError("Could not find data header row in file. Looking for columns with 'time', 'voltage', 'polarization'")
            
            # Use StringIO to read from specific line
            from io import StringIO
            data_text = ''.join(lines[header_row:])
            self.experimental_data = pd.read_csv(StringIO(data_text))
            
            print(f"Loaded columns: {list(self.experimental_data.columns)}")
            print(f"Data shape: {self.experimental_data.shape}")
            
            # Standardize column names - order matters!
            col_mapping = {}
            for col in self.experimental_data.columns:
                col_lower = col.lower().strip()
                # Check applied voltage FIRST before plain voltage
                if ('applied' in col_lower and 'voltage' in col_lower) and 'applied_voltage' not in col_mapping.values():
                    col_mapping[col] = 'applied_voltage'
                elif 'time' in col_lower and 'time' not in col_mapping.values():
                    col_mapping[col] = 'time'
                elif ('polarization' in col_lower or 'polar' in col_lower) and 'polarization' not in col_mapping.values():
                    col_mapping[col] = 'polarization'
                elif 'current' in col_lower and 'current' not in col_mapping.values():
                    col_mapping[col] = 'current'
                elif 'voltage' in col_lower and 'applied' not in col_lower and 'voltage' not in col_mapping.values():
                    col_mapping[col] = 'voltage'
            
            self.experimental_data = self.experimental_data.rename(columns=col_mapping)
            
            # Debug: print detected columns
            print(f"Original columns: {list(self.experimental_data.columns)}")
            print(f"Column mapping applied: {col_mapping}")
            
            # Check required columns exist
            required_cols = ['voltage', 'polarization']
            missing_cols = [c for c in required_cols if c not in self.experimental_data.columns]
            if missing_cols:
                raise ValueError(f"Could not find required columns: {missing_cols}\nDetected columns: {list(self.experimental_data.columns)}")
            
            # Update UI
            self.data_file_label.configure(text=os.path.basename(file_path))
            self.run_analysis_btn.configure(state="normal")
            self.fit_model_btn.configure(state="normal")
            
            # Initial plot
            self.update_analysis_graph()
            
            # Show data info
            self.analysis_results_text.delete("1.0", "end")
            info = f"""Data Loaded Successfully!

File: {os.path.basename(file_path)}
Points: {len(self.experimental_data)}
Columns: {list(self.experimental_data.columns)}

Voltage Range: {self.experimental_data['voltage'].min():.3f} to {self.experimental_data['voltage'].max():.3f} V
Polarization Range: {self.experimental_data['polarization'].min():.2f} to {self.experimental_data['polarization'].max():.2f} μC/cm²

Click 'Run Analysis' to extract loop parameters.
"""
            self.analysis_results_text.insert("1.0", info)
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load data:\n{str(e)}")
            
    def update_analysis_graph(self, *args):
        """Update the analysis graph."""
        self.analysis_ax.clear()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        graph_type = self.analysis_graph_type.get()
        
        if self.experimental_data is not None:
            # Prefer applied_voltage over voltage if available
            if 'applied_voltage' in self.experimental_data.columns:
                voltage = self.experimental_data['applied_voltage'].values
            else:
                voltage = self.experimental_data['voltage'].values
                
            polarization = self.experimental_data['polarization'].values
            
            if 'time' in self.experimental_data.columns:
                time = self.experimental_data['time'].values
            else:
                time = np.arange(len(voltage))
                
            if 'current' in self.experimental_data.columns:
                current = self.experimental_data['current'].values
            else:
                current = np.zeros_like(voltage)
            
            if graph_type == "P-V Loop":
                self.analysis_ax.plot(voltage, polarization, 'b-', linewidth=1.5, label='Experimental')
                if self.fitted_model is not None:
                    self.analysis_ax.plot(voltage, self.fitted_model['polarization'], 'r--', 
                                         linewidth=1.5, label='Model Fit')
                    self.analysis_ax.legend()
                self.analysis_ax.set_xlabel('Voltage (V)', fontsize=36)
                self.analysis_ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.analysis_ax.set_title('P-V Hysteresis Loop', fontsize=40)
                
            elif graph_type == "P-E Loop":
                film_thickness_nm = float(self.analysis_param_entries['a_film_thickness'].get())
                film_thickness = film_thickness_nm * 1e-9  # Convert nm to m
                E_field = (voltage / film_thickness) / 1e8  # Convert to MV/cm
                self.analysis_ax.plot(E_field, polarization, 'b-', linewidth=1.5)
                self.analysis_ax.set_xlabel('Electric Field (MV/cm)', fontsize=36)
                self.analysis_ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.analysis_ax.set_title('P-E Hysteresis Loop', fontsize=40)
                
            elif graph_type == "Current vs Time":
                time_us = time * 1e6
                current_uA = current * 1e6
                self.analysis_ax.plot(time_us, current_uA, 'r-', linewidth=1.5)
                self.analysis_ax.set_xlabel('Time (μs)', fontsize=36)
                self.analysis_ax.set_ylabel('Current (μA)', fontsize=36)
                self.analysis_ax.set_title('Current vs Time', fontsize=40)
                
            elif graph_type == "Current vs Voltage":
                current_uA = current * 1e6
                self.analysis_ax.plot(voltage, current_uA, 'r-', linewidth=1.5)
                self.analysis_ax.set_xlabel('Voltage (V)', fontsize=36)
                self.analysis_ax.set_ylabel('Current (μA)', fontsize=36)
                self.analysis_ax.set_title('Current vs Voltage', fontsize=40)
                
            elif graph_type == "Leakage Corrected P-V":
                if self.analysis_results is not None and 'P_corrected' in self.analysis_results:
                    self.analysis_ax.plot(voltage, polarization, 'b-', linewidth=1.5, 
                                         alpha=0.5, label='Original')
                    self.analysis_ax.plot(voltage, self.analysis_results['P_corrected'], 'g-', 
                                         linewidth=1.5, label='Leakage Corrected')
                    self.analysis_ax.legend()
                else:
                    self.analysis_ax.plot(voltage, polarization, 'b-', linewidth=1.5)
                self.analysis_ax.set_xlabel('Voltage (V)', fontsize=36)
                self.analysis_ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.analysis_ax.set_title('Leakage Corrected P-V Loop', fontsize=40)
                
            elif graph_type == "Model Fit Comparison":
                self.analysis_ax.plot(voltage, polarization, 'b-', linewidth=1.5, label='Experimental')
                if self.fitted_model is not None:
                    self.analysis_ax.plot(voltage, self.fitted_model['polarization'], 'r--', 
                                         linewidth=2, label='Landau Model Fit')
                    self.analysis_ax.legend()
                self.analysis_ax.set_xlabel('Voltage (V)', fontsize=36)
                self.analysis_ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
                self.analysis_ax.set_title('Model Fit Comparison', fontsize=40)
        else:
            self.analysis_ax.text(0.5, 0.5, 'Load experimental data to begin analysis', 
                                 ha='center', va='center', transform=self.analysis_ax.transAxes,
                                 fontsize=14, alpha=0.5)
            self.analysis_ax.set_xlabel('Voltage (V)', fontsize=36)
            self.analysis_ax.set_ylabel('Polarization (μC/cm²)', fontsize=36)
            self.analysis_ax.set_title('Analysis', fontsize=40)
        
        self.analysis_ax.tick_params(labelsize=36)
        self.analysis_ax.grid(True, alpha=0.3)
        self.analysis_ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.analysis_ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()
        
    def run_analysis(self):
        """Run comprehensive hysteresis analysis."""
        if self.experimental_data is None:
            messagebox.showwarning("No Data", "Please load experimental data first.")
            return
            
        try:
            # Prefer applied_voltage over voltage if available
            if 'applied_voltage' in self.experimental_data.columns:
                voltage = self.experimental_data['applied_voltage'].values
            else:
                voltage = self.experimental_data['voltage'].values
                
            polarization = self.experimental_data['polarization'].values
            
            if 'time' in self.experimental_data.columns:
                time = self.experimental_data['time'].values
            else:
                time = np.linspace(0, 1, len(voltage))
                
            if 'current' in self.experimental_data.columns:
                current = self.experimental_data['current'].values
            else:
                current = None
            
            # Get analysis parameters (convert from display units to SI)
            film_thickness_nm = float(self.analysis_param_entries['a_film_thickness'].get())
            film_thickness = film_thickness_nm * 1e-9  # Convert nm to m
            
            area_um2 = float(self.analysis_param_entries['a_area'].get())
            area = area_um2 * 1e-12  # Convert μm² to m²
            
            leakage_r = float(self.analysis_param_entries['a_leakage_r'].get())
            
            # Initialize results dictionary
            self.analysis_results = {}
            
            # =====================================================
            # 1. BASIC LOOP PARAMETERS
            # =====================================================
            
            # Find Pr+ and Pr- (polarization at V=0)
            zero_crossings_v = np.where(np.diff(np.sign(voltage)))[0]
            if len(zero_crossings_v) > 0:
                Pr_values = polarization[zero_crossings_v]
                Pr_plus = np.max(Pr_values)
                Pr_minus = np.min(Pr_values)
            else:
                Pr_plus = Pr_minus = 0
                
            # Find Vc+ and Vc- (voltage at P=0)
            zero_crossings_p = np.where(np.diff(np.sign(polarization)))[0]
            if len(zero_crossings_p) > 0:
                Vc_values = voltage[zero_crossings_p]
                Vc_plus = np.max(Vc_values)
                Vc_minus = np.min(Vc_values)
            else:
                Vc_plus = Vc_minus = 0
                
            Pmax = np.max(polarization)
            Pmin = np.min(polarization)
            Ps = (Pmax - Pmin) / 2  # Saturation polarization estimate
            
            # =====================================================
            # 2. IMPRINT ANALYSIS
            # =====================================================
            
            # Internal bias field (imprint)
            V_imprint = (Vc_plus + Vc_minus) / 2
            E_imprint = V_imprint / film_thickness
            
            # Polarization asymmetry
            P_asymmetry = (Pr_plus + Pr_minus) / 2  # Should be 0 for symmetric loop
            asymmetry_factor = abs(Pr_plus + Pr_minus) / (abs(Pr_plus) + abs(Pr_minus)) if (abs(Pr_plus) + abs(Pr_minus)) > 0 else 0
            
            # Coercive field asymmetry
            Ec_asymmetry = (abs(Vc_plus) - abs(Vc_minus)) / (abs(Vc_plus) + abs(Vc_minus)) if (abs(Vc_plus) + abs(Vc_minus)) > 0 else 0
            
            self.analysis_results['V_imprint'] = V_imprint
            self.analysis_results['E_imprint'] = E_imprint
            self.analysis_results['asymmetry_factor'] = asymmetry_factor
            
            # =====================================================
            # 3. LEAKAGE ANALYSIS
            # =====================================================
            
            if self.leakage_correction_var.get():
                # Estimate leakage contribution
                # P_leak = integral(V/R * dt) / Area
                dt = np.mean(np.diff(time)) if len(time) > 1 else 1e-6
                leakage_current = voltage / leakage_r
                P_leak = np.cumsum(leakage_current) * dt / area * 1e4  # Convert to μC/cm²
                
                # Subtract leakage
                P_corrected = polarization - P_leak
                P_corrected = P_corrected - np.mean(P_corrected)  # Re-center
                
                self.analysis_results['P_corrected'] = P_corrected
                self.analysis_results['P_leakage'] = P_leak
            else:
                self.analysis_results['P_corrected'] = polarization
                
            # =====================================================
            # 4. DEAD LAYER ANALYSIS
            # =====================================================
            
            # Calculate effective capacitance from slope at coercive field
            # C_eff = dP/dV at V=Vc
            epsilon_eff = None
            slope_at_Vc = None
            
            if len(zero_crossings_p) > 0:
                try:
                    # Get slope near coercive field
                    idx_vc = zero_crossings_p[0]
                    window = 20  # Larger window for more stable fit
                    start_idx = max(0, idx_vc - window)
                    end_idx = min(len(voltage), idx_vc + window)
                    
                    if end_idx - start_idx > 5:
                        v_window = voltage[start_idx:end_idx]
                        p_window = polarization[start_idx:end_idx]
                        
                        # Check if we have enough variation in the data
                        v_range = np.max(v_window) - np.min(v_window)
                        p_range = np.max(p_window) - np.min(p_window)
                        
                        if v_range > 1e-10 and p_range > 1e-10:
                            # Use numpy's lstsq which is more robust
                            A = np.vstack([v_window, np.ones(len(v_window))]).T
                            result = np.linalg.lstsq(A, p_window, rcond=None)
                            slope = result[0][0]
                            
                            # Convert slope to permittivity: dP/dE = epsilon_0 * epsilon_r
                            # dP/dV = dP/dE * dE/dV = epsilon_0 * epsilon_r / d
                            epsilon_0 = 8.854e-12
                            epsilon_eff = slope * 1e-6 * film_thickness / epsilon_0  # Convert μC/cm² to C/m²
                            slope_at_Vc = slope
                            
                            self.analysis_results['epsilon_eff'] = epsilon_eff
                            self.analysis_results['slope_at_Vc'] = slope_at_Vc
                except Exception as e:
                    print(f"Dead layer analysis failed: {e}")
                    # Continue without dead layer analysis
                    
            # =====================================================
            # 5. LOOP AREA (Energy)
            # =====================================================
            
            # Calculate loop area (energy density)
            loop_area = np.abs(np.trapz(polarization, voltage))  # μC/cm² * V = μJ/cm²
            self.analysis_results['loop_area'] = loop_area
            
            # Enable save button
            self.save_analysis_btn.configure(state="normal")
            
            # Update results display
            self.analysis_results_text.delete("1.0", "end")
            
            results = f"""═══════════════════════════════════════
       HYSTERESIS LOOP ANALYSIS
═══════════════════════════════════════

BASIC LOOP PARAMETERS
─────────────────────────────────────
  Pr+:  {Pr_plus:.2f} μC/cm²
  Pr-:  {Pr_minus:.2f} μC/cm²
  Vc+:  {Vc_plus:.3f} V
  Vc-:  {Vc_minus:.3f} V
  Ps:   {Ps:.2f} μC/cm²
  Pmax: {Pmax:.2f} μC/cm²
  Pmin: {Pmin:.2f} μC/cm²

IMPRINT ANALYSIS
─────────────────────────────────────
  Built-in Voltage: {V_imprint*1000:.2f} mV
  Built-in Field:   {E_imprint/1e6:.2f} MV/m
  P Asymmetry:      {P_asymmetry:.2f} μC/cm²
  Asymmetry Factor: {asymmetry_factor:.3f}
  Vc Asymmetry:     {Ec_asymmetry:.3f}

ENERGY ANALYSIS
─────────────────────────────────────
  Loop Area: {loop_area:.2f} μJ/cm²
"""
            
            if 'epsilon_eff' in self.analysis_results:
                results += f"""
DEAD LAYER ANALYSIS
─────────────────────────────────────
  Slope at Vc: {self.analysis_results['slope_at_Vc']:.2f} (μC/cm²)/V
  ε_eff:       {self.analysis_results['epsilon_eff']:.1f}
"""
            
            if self.leakage_correction_var.get():
                results += f"""
LEAKAGE CORRECTION
─────────────────────────────────────
  Leakage R:  {leakage_r:.2e} Ω
  Correction: Applied
"""
            
            self.analysis_results_text.insert("1.0", results)
            self.update_analysis_graph()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def auto_estimate_imprint(self):
        """Auto-estimate imprint and offset parameters from experimental data."""
        if self.experimental_data is None:
            messagebox.showwarning("No Data", "Please load experimental data first.")
            return
            
        try:
            # Get voltage and polarization
            if 'applied_voltage' in self.experimental_data.columns:
                voltage = self.experimental_data['applied_voltage'].values
            else:
                voltage = self.experimental_data['voltage'].values
            polarization = self.experimental_data['polarization'].values
            
            # Estimate P_offset as the center of the polarization range
            P_offset = (np.max(polarization) + np.min(polarization)) / 2
            
            # Estimate V_imprint from the coercive voltages
            # Center the polarization first
            P_centered = polarization - P_offset
            
            # Find where P crosses zero
            zero_crossings = np.where(np.diff(np.sign(P_centered)))[0]
            
            if len(zero_crossings) >= 2:
                Vc_values = voltage[zero_crossings]
                Vc_plus = np.max(Vc_values)
                Vc_minus = np.min(Vc_values)
                V_imprint = (Vc_plus + Vc_minus) / 2
            else:
                V_imprint = 0
            
            # Update the entry fields
            self.analysis_param_entries['a_v_imprint'].delete(0, 'end')
            self.analysis_param_entries['a_v_imprint'].insert(0, f"{V_imprint:.3f}")
            
            self.analysis_param_entries['a_p_offset'].delete(0, 'end')
            self.analysis_param_entries['a_p_offset'].insert(0, f"{P_offset:.2f}")
            
            # Show info
            messagebox.showinfo("Auto-Estimate Complete", 
                f"Estimated parameters:\n\n"
                f"V_imprint: {V_imprint*1000:.1f} mV\n"
                f"P_offset: {P_offset:.2f} μC/cm²\n\n"
                f"These values have been filled in.\n"
                f"Adjust as needed and click 'Fit Landau Model'.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Auto-estimate failed:\n{str(e)}")

    def fit_landau_model(self):
        """Fit Landau model to experimental data with imprint and dead layer effects."""
        if self.experimental_data is None:
            messagebox.showwarning("No Data", "Please load experimental data first.")
            return
            
        try:
            # Prefer applied_voltage over voltage if available
            if 'applied_voltage' in self.experimental_data.columns:
                voltage = self.experimental_data['applied_voltage'].values
            else:
                voltage = self.experimental_data['voltage'].values
                
            polarization = self.experimental_data['polarization'].values
            
            # Get fitting parameters (convert units)
            film_thickness_nm = float(self.analysis_param_entries['a_film_thickness'].get())
            film_thickness = film_thickness_nm * 1e-9  # Convert nm to m
            
            area_um2 = float(self.analysis_param_entries['a_area'].get())
            area = area_um2 * 1e-12  # Convert μm² to m²
            
            temperature = float(self.analysis_param_entries['a_temperature'].get())
            epsilon_r = float(self.analysis_param_entries['a_epsilon_r'].get())
            
            # Get imprint and dead layer parameters
            V_imprint = float(self.analysis_param_entries['a_v_imprint'].get())
            P_offset_uC = float(self.analysis_param_entries['a_p_offset'].get())
            P_offset = P_offset_uC / 100  # Convert μC/cm² to C/m²
            
            dead_layer_nm = float(self.analysis_param_entries['a_dead_layer'].get())
            dead_layer_thickness = dead_layer_nm * 1e-9  # Convert nm to m
            dead_layer_epsilon = float(self.analysis_param_entries['a_dead_epsilon'].get())
            
            # Landau coefficients
            a0_init = float(self.analysis_param_entries['a_fit_a0'].get())
            b_init = float(self.analysis_param_entries['a_fit_b'].get())
            c_init = float(self.analysis_param_entries['a_fit_c'].get())
            T0_init = float(self.analysis_param_entries['a_fit_T0'].get())
            
            # Get electrode parameters from selection
            elec_mat = self.analysis_electrode.get()
            elec_params = get_electrode_params(elec_mat) if elec_mat != "Other" else None
            screening_lambda = elec_params['screening_lambda'] if elec_params else 0.1e-9
            permittivity_e = elec_params['permittivity_e'] if elec_params else 1e5
            
            # Get substrate parameters
            sub_mat = self.analysis_substrate.get()
            sub_params = get_substrate_params(sub_mat) if sub_mat != "Other" else None
            sub_lattice = sub_params['lattice_a'] if sub_params else 0.395e-9
            
            # Get FE parameters
            fe_mat = self.analysis_fe_material.get()
            fe_params = get_ferroelectric_params(fe_mat) if fe_mat != "Other" else None
            fe_lattice = fe_params['lattice_a'] if fe_params else 0.4e-9
            Q12 = fe_params['Q12'] if fe_params else -0.045
            s11 = fe_params['s11'] if fe_params else 14e-12
            s12 = fe_params['s12'] if fe_params else -4.5e-12
            
            # Build material dict for fitting
            material_dict = {
                'ferroelectric': {
                    'a0': a0_init,
                    'b': b_init,
                    'c': c_init,
                    'T0': T0_init,
                    'Q12': Q12,
                    's11': s11,
                    's12': s12,
                    'lattice_a': fe_lattice,
                    'film_thickness': film_thickness,
                    'epsilon_r': epsilon_r,
                    'leakage_resistance': 1e100,
                },
                'substrate': {'lattice_a': sub_lattice},
                'electrode': {
                    'screening_lambda': screening_lambda,
                    'permittivity_e': permittivity_e,
                    'area': area
                }
            }
            
            # Run model with imprint and dead layer effects
            model = FerroelectricModel(material_dict, temperature=temperature)
            P_model = model.run_hysteresis_with_imprint_and_dead_layer(
                voltage, 
                temperature=temperature,
                V_imprint=V_imprint,
                dead_layer_thickness=dead_layer_thickness,
                dead_layer_epsilon=dead_layer_epsilon,
                P_offset=P_offset
            )
            P_model_uC = P_model * 100  # Convert to μC/cm²
            
            self.fitted_model = {
                'voltage': voltage,
                'polarization': P_model_uC,
                'parameters': {
                    'a0': a0_init,
                    'b': b_init,
                    'c': c_init,
                    'T0': T0_init,
                    'V_imprint': V_imprint,
                    'P_offset': P_offset_uC,
                    'dead_layer': dead_layer_nm,
                    'dead_layer_epsilon': dead_layer_epsilon
                }
            }
            
            # Calculate fit quality
            residuals = polarization - P_model_uC
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((polarization - np.mean(polarization))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Calculate model loop parameters
            model_Pr_plus = np.max(P_model_uC[np.where(np.diff(np.sign(voltage)))[0]]) if len(np.where(np.diff(np.sign(voltage)))[0]) > 0 else 0
            model_Pr_minus = np.min(P_model_uC[np.where(np.diff(np.sign(voltage)))[0]]) if len(np.where(np.diff(np.sign(voltage)))[0]) > 0 else 0
            
            # Update results
            current_text = self.analysis_results_text.get("1.0", "end")
            fit_results = f"""
═══════════════════════════════════════
       LANDAU MODEL FIT
═══════════════════════════════════════
  Material: {fe_mat}
  Substrate: {sub_mat}
  Electrode: {elec_mat}
  
DEVICE PARAMETERS
─────────────────────────────────────
  Film: {film_thickness_nm:.1f} nm
  Area: {area_um2:.1f} μm²
  Temperature: {temperature:.1f} K
  
FIT QUALITY
─────────────────────────────────────
  R²:   {r_squared:.4f}
  RMSE: {rmse:.2f} μC/cm²

IMPRINT & DEAD LAYER
─────────────────────────────────────
  V_imprint:    {V_imprint*1000:.1f} mV
  P_offset:     {P_offset_uC:.2f} μC/cm²
  Dead Layer:   {dead_layer_nm:.2f} nm
  Dead Layer ε: {dead_layer_epsilon:.1f}
  
LANDAU COEFFICIENTS
─────────────────────────────────────
  a₀: {a0_init:.2e} J·m/(C²·K)
  b:  {b_init:.2e} J·m⁵/C⁴
  c:  {c_init:.2e} J·m⁹/C⁶
  T₀: {T0_init:.1f} K

INTERPRETATION
─────────────────────────────────────
"""
            # Add interpretation based on parameters
            if abs(V_imprint) > 0.1:
                fit_results += f"  • Significant imprint detected ({V_imprint*1000:.0f} mV)\n"
                fit_results += f"    Suggests asymmetric electrode/interface\n"
            
            if abs(P_offset_uC) > 5:
                fit_results += f"  • Large P offset ({P_offset_uC:.1f} μC/cm²)\n"
                fit_results += f"    May indicate trapped charge or\n"
                fit_results += f"    preferential domain orientation\n"
            
            if dead_layer_nm > 0.5:
                fit_results += f"  • Dead layer: {dead_layer_nm:.1f} nm\n"
                fit_results += f"    Non-ferroelectric interface layer\n"
                fit_results += f"    causes loop tilt/tilted saturation\n"
            
            self.analysis_results_text.delete("1.0", "end")
            self.analysis_results_text.insert("1.0", current_text.strip() + fit_results)
            
            # Update graph to show fit
            self.analysis_graph_type.set("Model Fit Comparison")
            self.update_analysis_graph()
            
        except Exception as e:
            messagebox.showerror("Fitting Error", f"Model fitting failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def save_analysis(self):
        """Save analysis results to file."""
        if self.analysis_results is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Analysis Results"
        )
        
        if file_path:
            try:
                # Save results text
                results_text = self.analysis_results_text.get("1.0", "end")
                with open(file_path, 'w') as f:
                    f.write(results_text)
                    
                # Also save corrected data if available
                if 'P_corrected' in self.analysis_results:
                    csv_path = file_path.replace('.txt', '_corrected_data.csv')
                    df = pd.DataFrame({
                        'voltage': self.experimental_data['voltage'],
                        'polarization_original': self.experimental_data['polarization'],
                        'polarization_corrected': self.analysis_results['P_corrected']
                    })
                    df.to_csv(csv_path, index=False)
                    
                current_text = self.analysis_results_text.get("1.0", "end")
                self.analysis_results_text.delete("1.0", "end")
                self.analysis_results_text.insert("1.0", current_text.strip() + f"\n\nResults saved to:\n{os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")

    # =========================================================================
    # MAIN
    # =========================================================================
        
    def run(self):
        """Start the application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = FerroelectricSimulator()
    app.run()