# Ferroelectric Hysteresis Simulator

A Python application for simulating ferroelectric hysteresis loops using Landau-Devonshire theory.

## Project Structure

```
ferroelectric_app/
├── main.py              # Entry point - run this to start the app
├── gui.py               # Main GUI application (CustomTkinter)
├── fe_model.py          # Ferroelectric physics model
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python main.py
```

Or alternatively:
```bash
python gui.py
```

## Usage

### Simulation Tab

1. **Select Materials**: Choose the ferroelectric material (PZT, HZO), substrate (SRO, SI), and electrode (YBCO, Pt) from the dropdowns. Select "Other" to manually enter parameters.

2. **Set Waveform Parameters**:
   - Waveform Type: Triangle or Sine
   - Amplitude: Peak voltage (V)
   - Frequency: Measurement frequency (Hz)
   - Cycles: Number of complete cycles
   - Temperature: Operating temperature (K)

3. **View/Edit Device Parameters**: All Landau coefficients, lattice parameters, and material properties can be viewed and edited in the side panel.

4. **Run Simulation**: Click "Run Simulation" to execute the Landau-Devonshire model.

5. **View Results**: 
   - Switch between P-E Loop, P-V Loop, and Applied Voltage graphs
   - Loop parameters (Pr, Vc, etc.) are shown in the Results section

## Model Description

The simulation uses Landau-Devonshire theory with:

- **Strain effects**: From lattice mismatch between film and substrate
- **Depolarization effects**: From incomplete screening at electrodes
- **Parasitic effects** (optional): Linear dielectric contribution and leakage current

### Key Equations

The free energy expansion:
```
G = a₀(T-T₀)P² + bP⁴ + cP⁶ - EP
```

With renormalized coefficients accounting for:
- Epitaxial strain: `η_m = (a_sub - a_fe) / a_fe`
- Depolarization field from electrode screening

## File Descriptions

### `fe_model.py`
Contains the `FerroelectricModel` class with methods:
- `run_landau_hysteresis_simulation()`: Core Landau-Devonshire solver
- `add_parasitic_effects()`: Adds dielectric and leakage contributions
- `run_simulation()`: Complete simulation wrapper
- `calculate_loop_parameters()`: Extract Pr, Vc, etc.

### `gui.py`
Contains the `FerroelectricSimulator` class:
- Material database with preset parameters
- Parameter input/editing interface
- Matplotlib plotting integration
- Results display

## Adding New Materials

To add a new ferroelectric material, add an entry to the `materials` dictionary in `gui.py`:

```python
'NewMaterial': {
    'ferroelectric': {
        'a0': ...,      # Landau coefficient (J·m/C²·K)
        'b': ...,       # Landau coefficient (J·m⁵/C⁴)
        'c': ...,       # Landau coefficient (J·m⁹/C⁶)
        'T0': ...,      # Curie-Weiss temperature (K)
        'Q12': ...,     # Electrostriction coefficient (m⁴/C²)
        's11': ...,     # Elastic compliance (m²/N)
        's12': ...,     # Elastic compliance (m²/N)
        'lattice_a': ...,       # Lattice parameter (m)
        'film_thickness': ...,  # Film thickness (m)
        'epsilon_r': ...,       # Relative permittivity
        'leakage_resistance': ...,  # Leakage resistance (Ω)
    },
    'substrate': {
        'lattice_a': ...  # Substrate lattice parameter (m)
    },
    'electrode': {
        'screening_lambda': ...,  # Screening length (m)
        'permittivity_e': ...,    # Electrode permittivity
        'area': ...               # Device area (m²)
    }
}
```

## Future Development

- [ ] Analysis tab for experimental data fitting
- [ ] Export simulation results to CSV
- [ ] Multiple loop comparison
- [ ] Temperature sweep simulations
- [ ] Fatigue modeling
