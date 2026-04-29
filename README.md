# FeSim: An Open-Source Software for Ferroelectric Hysteresis Modeling and Analysis

**Author:** Rohan S. Pankaj  
**Advisor:** Prof. Lucas Caretta  
**Engineering Senior Capstone — Brown University**

---

## Overview

Ferroelectric materials maintain spontaneous electric polarization without an external field. Applying a field switches this polarization, enabling multiple stable states — the basis of technologies like FeRAM. These devices rely on ferroelectric capacitors (FeCAPs), where a ferroelectric film replaces the dielectric in a standard capacitor. Hysteresis measurements of FeCAPs visualize this switching behavior and expose key material properties: saturation polarization (*P*s), remanent polarization (*P*R), and coercive field (*E*c).

In practice, measured loops are distorted by nonidealities — leakage current, built-in fields (imprint), dead layers, and dielectric tilt. FeSim is open-source software that simulates ferroelectric hysteresis and these nonidealities, designed to help researchers identify sources of distortion in their devices and inform material growth and iteration.

---

## Features

- **Forward simulation** — generate P-E and P-V loops from Landau-Ginzburg-Devonshire theory for HZO, PZT, or BaTiO₃
- **Parasitic effects** — dielectric response, ohmic leakage, dead layers, and imprint
- **Experimental analysis** — load a probe-station CSV (PIEC-compatible), extract P_r, V_c, loop area, imprint, and effective permittivity
- **Landau model overlay** — superimpose a physics model on experimental data with adjustable dead layer and imprint parameters
- **Material database** — literature-calibrated parameters for HZO, PZT, BTO; substrates SRO, STO, Si, MgO, TiN; electrodes Pt, TiN, W, YBCO, SRO

---

## Installation

```bash
cd ferroelectric_app
pip install -r requirements.txt
python main.py
```

---

## Physics and Mathematics

The simulation follows the methodology outlined in Chandra et al. [1].

### 1. Free Energy and the Constitutive Relation

The free energy density of the ferroelectric is modeled as a Landau-Ginzburg-Devonshire expansion in the out-of-plane polarization *P*:

$$\mathcal{F}_P = \frac{1}{2}aP^2 + \frac{1}{4}bP^4 + \frac{1}{6}cP^6 - EP$$

where *E* is the applied electric field. Minimizing with respect to *P* (i.e. setting ∂𝒻_P/∂P = 0) yields the constitutive relation used in the simulation:

$$E = aP + bP^3 + cP^5 \tag{3}$$

This is a fifth-degree polynomial in *P*, solved numerically at each voltage step using `scipy.fsolve`. The double-well shape of the free energy produces two stable polarization states — the physical origin of hysteresis.

---

### 2. Landau Coefficients

The three Landau coefficients each capture a distinct physical effect.

**Coefficient *a* — temperature sensitivity:**

$$a = a_0(T - T_0) \tag{4}$$

where *T*₀ is the Curie-Weiss temperature and *a*₀ is a material-specific proportionality constant. For *T* < *T*₀, *a* < 0, giving a double-well (ferroelectric phase). For *T* > *T*₀, *a* > 0, giving a single-well (paraelectric phase).

**Coefficient *b* — fourth-order stiffness:**  
Controls the magnitude of spontaneous polarization. Its sign determines the order of the phase transition: *b* < 0 is first-order (BaTiO₃), *b* > 0 is second-order (HZO, PZT).

**Coefficient *c* — stabilization term:**  
Ensures the free energy is bounded from below. Always positive.

---

### 3. Thin-Film Renormalization: Misfit Strain

Bulk Landau coefficients must be corrected for epitaxial thin films, where biaxial strain from lattice mismatch with the substrate alters the ferroelectric behavior. The misfit strain is:

$$\eta = \frac{a_\text{substrate} - a_\text{FE}}{a_\text{substrate}} \tag{5}$$

where $a_\text{substrate}$ and $a_\text{FE}$ are the in-plane lattice constants. Via electrostrictive coupling (*Q*₁₂) and elastic compliance (*s*₁₁, *s*₁₂), the coefficients *a* and *b* are renormalized:

$$\tilde{a} = a - \frac{4\eta Q_{12}}{s_{11} + s_{12}} \tag{6}$$

$$\tilde{b} = b - \frac{4Q_{12}^2}{s_{11} + s_{12}} \tag{7}$$

The coefficient *c* is unaffected by strain. The misfit strain also shifts the effective Curie-Weiss temperature:

$$T^* = T_0 + \frac{4\eta Q_{12}}{a_0(s_{11} + s_{12})} \tag{8}$$

The final equation solved by the simulation is:

$$E = \tilde{a}P + \tilde{b}P^3 + cP^5 \tag{9}$$

---

### 4. Hysteresis Loop Tracing

Equation (9) is multivalued in the switching region — for a given *E* there are three solutions for *P*, of which two are stable (the upper and lower branches of the loop). The simulation traces the physically realized metastable curve:

- **Upper branch** — the solver follows the high-*P* root until *V* falls below the negative coercive voltage *V*c⁻
- **Lower branch** — the solver follows the low-*P* root until *V* rises above the positive coercive voltage *V*c⁺

At each step `scipy.fsolve` solves V(P) - V_target = 0 with the current branch as the initial guess. When the voltage crosses a coercive point, the initial guess is switched to the opposite branch and the solver finds the new stable solution.

The coercive voltages are found analytically from the spinodal condition d*V*/d*P* = 0:

$$5cP^4 + 3\tilde{b}P^2 + \tilde{a} = 0$$

which is a quadratic in *P*² with roots at the switching polarizations.

---

### 5. Parasitic Effects

Real loops are distorted by currents beyond the ideal switching current. These are modeled by augmenting the total current:

$$I_\text{total} = I_\text{ideal switching} + I_\text{dielectric} + I_\text{leakage} \tag{10}$$

#### 5.1 Dielectric Response

The non-switching dielectric background of the ferroelectric contributes:

$$I_\text{dielectric} = \text{Area} \times \frac{\varepsilon_0 \varepsilon_r}{d} \times \frac{dV}{dt} \tag{11}$$

where *d* is the film thickness and ε*r* is the background permittivity. Integrating this current into polarization produces a linear tilt of the P-V loop.

#### 5.2 Ohmic Leakage

Leakage current is calculated from the applied voltage and a user-specified leakage resistance:

$$I_\text{leakage} = \frac{V}{R_\text{leak}}$$

Integrating this into the polarization fattens the loop and inflates the apparent loop area without contributing to true switchable polarization.

#### 5.3 Dead Layer

A non-ferroelectric interfacial layer (thickness *t*dl, permittivity ε*dl*) acts as a series capacitor, taking a fraction of the applied voltage:

$$V_\text{total} = V_\text{FE} + V_\text{dead} = V_\text{FE} + \frac{P \cdot t_\text{dl}}{\varepsilon_0 \varepsilon_\text{dl}}$$

This tilts and narrows the loop, reducing apparent *P*r and *E*c.

#### 5.4 Imprint

Asymmetric charge trapping at one interface generates a built-in field that shifts the loop horizontally. In the simulation, this is applied as:

$$V_\text{eff} = V_\text{applied} - V_\text{imprint}$$

Imprint is extracted from data as the average of the positive and negative coercive voltages:

$$V_\text{imprint} = \frac{V_c^+ + V_c^-}{2}$$

---

### 6. Leakage Correction

Users can manually correct leakage in the analysis tab by specifying a leakage resistance. The estimated leakage current is calculated and subtracted from the current response. Automatic leakage extraction from multi-frequency measurements (following Meyer et al. [2]) is a planned future feature.

---

### 7. Loop Parameter Extraction

From a measured or simulated P-V loop:

| Parameter | Definition | Physical meaning |
|---|---|---|
| *P*r⁺, *P*r⁻ | Polarization at *V* = 0 | Remanent (stored) polarization |
| *V*c⁺, *V*c⁻ | Voltage at *P* = 0 | Coercive voltages |
| *P*s | (*P*max − *P*min) / 2 | Saturation polarization estimate |
| *V*imprint | (*V*c⁺ + *V*c⁻) / 2 | Built-in bias from interface asymmetry |
| Loop area | ∮ *P* d*V* | Energy dissipated per cycle (μJ/cm²) |
| ε_eff | (d*P*/d*V*) · *t* / ε₀ at *V*c | Effective permittivity near switching |

---

### 8. Material Parameters

Coefficients are taken from literature on bulk properties and corrected for thin films per equations (5)–(8).

| Parameter | HZO | PZT (53/47) | BaTiO₃ | Units |
|---|---|---|---|---|
| *a*₀ | 4.0×10⁶ | 3.1×10⁵ | 4.1×10⁵ | J·m/(C²·K) |
| *b* | 4.74×10¹⁰ | 1.91×10⁸ | −2.1×10⁸ | J·m⁵/C⁴ |
| *c* | 1.0×10¹⁰ | 1.40×10⁹ | 1.3×10⁹ | J·m⁹/C⁶ |
| *T*₀ | 773 K | 673 K | 393 K | K |
| *Q*₁₂ | −0.02 | −0.046 | −0.034 | m⁴/C² |
| ε*r* | 25 | 500 | 1500 | — |
| Film thickness | 10 nm | 40 nm | 100 nm | — |

BTO has *b* < 0, making it a first-order (discontinuous) ferroelectric phase transition. HZO coefficients are calibrated to Tian et al. [9].

---

## Design Decisions

**Python** was chosen for its scientific computing ecosystem (NumPy, SciPy, Matplotlib, Pandas) and suitability for rapid prototyping in research environments. `scipy.fsolve` is used to numerically solve equation (9), avoiding the need for a closed-form analytical solution and preserving computational resources.

**LGD model over alternatives** — the fifth-order polynomial is computationally lightweight, well-suited for a desktop app, and physically intuitive. The key limitation is that it cannot capture frequency-dependent switching dynamics.

**Fitting** is deliberately constrained to *a*₀, *b*, *c*, and *T*₀ only, to prevent overfitting when dozens of parameters are present.

**PIEC compatibility** — data files accepted by FeSim follow the CSV structure produced by [PIEC](https://github.com/ElPsyKurisu/piec), an open-source ferroelectric testing tool. The shared Python/Tkinter foundation enables future integration.

---

## Project Structure

```
ferroelectric_app/
├── main.py                  # Entry point
├── fe_model.py              # LGD physics engine
├── gui.py                   # CustomTkinter GUI (Simulation + Analysis tabs)
├── material_parameters.py   # Material, substrate, and electrode databases
└── requirements.txt
```

---

## Future Work

- Automatic fitting of Landau coefficients to experimental data
- Multi-file upload for frequency-dependent leakage extraction (following Meyer et al. [2])

---

## References

[1] P. Chandra and P. B. Littlewood, "A Landau Primer for Ferroelectrics," in *Physics of Ferroelectrics*, Springer, Berlin, 2007.

[2] R. Meyer and R. Waser, "Dynamic leakage current compensation in ferroelectric thin-film capacitor structures," *Applied Physics Letters*, vol. 86, 2005.

[3] C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[4] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.

[5] P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020.

[6] T. Schimansky, "CustomTkinter," 2024. https://github.com/TomSchimansky/CustomTkinter

[7] A. Fratian et al., "piec," GitHub, 2025. https://github.com/ElPsyKurisu/piec

[9] Tian et al., *Phys. Rev. Applied* **20**, 054007 (2023) — HZO Landau coefficients.
