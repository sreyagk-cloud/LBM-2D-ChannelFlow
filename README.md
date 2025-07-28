
# 🌀 2D LBM Channel Flow Simulation (D2Q9)  
### A High-Performance Replication Based on Sukop & Thorne’s Paper  
**Paper Reference**: [A simple Lattice Boltzmann method for simulating fluid flows ](https://www.researchgate.net/publication/253259723_Lattice_Boltzmann_Modeling_An_Introduction_for_Geoscientists_and_Engineers)

---

## 📌 Abstract  
This project replicates the classical 2D channel flow simulation using the **Lattice Boltzmann Method (LBM)** with the D2Q9 lattice model. Implemented in Python and optimized using **Numba** for performance, this study compares numerical results with the analytical solution of fully developed Poiseuille flow and includes multiple performance and visualization metrics.

---

## 🎯 Goals  
- Implement LBM D2Q9 scheme for 2D incompressible channel flow  
- Validate against analytical Poiseuille velocity profile  
- Record convergence, error metrics, and flow field evolution  
- Visualize intermediate and final states  
- Benchmark performance using Numba and array operations  

---

## 🧠 Original Paper Overview  
**Sukop & Thorne (2010)**:  
They propose a minimal LBM implementation to simulate fluid flow in a 2D channel with bounce-back boundary conditions and a body force. The final velocity profile is compared with the analytical solution of the Navier-Stokes equations.

---

## 🛠️ Project Structure  

```
├── lbm_flow.py              # Core LBM time-stepping + collision + streaming
├── Create_lbm_channel_flow.py # Initializes lattice and calls solver
├── compare_to_analytical.py   # Compares final velocity to analytical solution
├── velocity_animation.py      # Creates .mp4 animation of ux field
├── velocity_contour.py        # Generates final velocity contour plot
├── inspect_ux.py              # For inspecting intermediate or mid-channel ux
├── check_convergence.py       # Residual/error tracking across time
├── Mid_channel.py             # Plots centerline velocity over time
├── output/
│   ├── ux_final.npy
│   ├── uy_final.npy
│   ├── ux_history.npz
│   ├── ux_snapshots.npz
│   ├── residuals.csv
│   ├── ux_animation.mp4
│   ├── Mid channel plot.png
│   ├── Velocity contour.png
│   ├── convergence_plot.png
│   ├── ...
```

---

## 📊 Output Visualizations  

### ✅ 1. Final Velocity Profile Comparison  
![Mid Channel Plot](output\Velocity_profile_comparison.png)

### 🎥 2. Velocity Animation  
🎞️ Saved as: `output\ux_animation.mp4`

### 🌊 3. Velocity Contour Plot  
![Velocity Contour](output\velocity_contour.png)

### 📉 4. Convergence Plot  
![Convergence Plot](./output/convergence_plot.png)

### 📉 5. Mid Channel Velocity Contour  
![Mid Channel Velocity contour](output\mid_channel.png)

---

## 📈 Performance Benchmarks  

| Feature               | Value                   |
|----------------------|-------------------------|
| Grid Size            | 100 × 50                |
| Time Steps           | 10000                   |
| Execution Time       | ~12 seconds (Numba JIT) |
| Acceleration Used    | Numba + Array Ops       |
| Memory Usage         | < 250 MB                |

---

## 📐 Analytical Validation  
The velocity profile \( u_x(y) \) is compared with the analytical Poiseuille flow:  
\[
u_x(y) = \frac{4U_{max}}{H^2} y (H - y)
\]  
The match is near-perfect after steady-state convergence.

---

## 💡 Replication Notes  
- Used Numba’s `@njit` and parallel-friendly operations for speedup  
- Avoided `np.roll` in inner loops for full compatibility with Numba  
- Compared `ux_final` normalized against analytical max velocity  
- Snapshots and `ux_history` are saved for post-analysis  

---

## 🧪 How to Run  

```bash
# 1. Run the solver
python Create_lbm_channel_flow.py

# 2. Visualize Results
python compare_to_analytical.py
python velocity_contour.py
python velocity_animation.py
python Mid_channel.py
python check_convergence.py
```

---

## 📬 Contact  
Feel free to connect for questions, collaboration or improvements!
