import numpy as np
import os
import csv
from numba import njit, prange

# --- Simulation parameters ---
nx, ny = 400, 100               # Grid size
tau = 0.6                       # Relaxation time
u_max = 0.1                     # Max inlet velocity
n_steps = 100000                # Max iterations
save_interval = 100            # Save every N steps
convergence_tol = 1e-12        # Convergence threshold
output_dir = "output"          # Output folder

os.makedirs(output_dir, exist_ok=True)

# --- Lattice weights and directions (D2Q9) ---
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1],
              [1,1], [-1,1], [-1,-1], [1,-1]])
cx = np.ascontiguousarray(c[:, 0].reshape((9, 1, 1)))
cy = np.ascontiguousarray(c[:, 1].reshape((9, 1, 1)))

@njit
def equilibrium(rho, ux, uy, w, c):
    feq = np.zeros((9, ny, nx))
    for i in range(9):
        cu = c[i, 0] * ux + c[i, 1] * uy
        feq[i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*(ux**2 + uy**2))
    return feq

@njit(parallel=True)
def stream(f, c):
    nx, ny = f.shape[1], f.shape[2]
    f_streamed = np.empty_like(f)
    for i in prange(9):
        dx, dy = c[i]
        for x in range(nx):
            for y in range(ny):
                x_from = (x - dx) % nx
                y_from = (y - dy) % ny
                f_streamed[i, x, y] = f[i, x_from, y_from]
    f[:] = f_streamed

@njit
def compute_macros(f, cx, cy):
    rho = np.sum(f, axis=0)
    ux = np.sum(f * cx, axis=0) / rho
    uy = np.sum(f * cy, axis=0) / rho
    return rho, ux, uy

def run_simulation():
    f = np.ones((9, ny, nx))
    rho, ux, uy = compute_macros(f, cx, cy)
    prev_ux = ux.copy()

    saved_ux = []
    steps = []
    residuals = []
    ux_history = []

    for step in range(n_steps):
        rho, ux, uy = compute_macros(f, cx, cy)

        # Inlet BC — parabolic Poiseuille profile
        y = np.arange(ny)
        uy[:, 0] = 0
        ux[:, 0] = 4 * u_max * (y / (ny - 1)) * (1 - y / (ny - 1))
        rho[:, 0] = (f[0, :, 0] + f[2, :, 0] + f[4, :, 0] +
                     2 * (f[3, :, 0] + f[6, :, 0] + f[7, :, 0])) / (1 - ux[:, 0])

        # Collision
        feq = equilibrium(rho, ux, uy, w, c)
        f += -(f - feq) / tau

        # Streaming
        stream(f, c)

        # Bounce-back: Top and Bottom walls
        f[2, ny-1, :] = f[4, ny-1, :]
        f[5, ny-1, :] = f[7, ny-1, :]
        f[6, ny-1, :] = f[8, ny-1, :]

        f[4, 0, :] = f[2, 0, :]
        f[7, 0, :] = f[5, 0, :]
        f[8, 0, :] = f[6, 0, :]

        # Outlet: Neumann condition (copy last column)
        f[:, :, -1] = f[:, :, -2]

        if step % save_interval == 0:
            residual = np.linalg.norm(ux - prev_ux) / np.linalg.norm(prev_ux)
            steps.append(step)
            residuals.append(residual)
            print(f"Step {step} | Residual: {residual:.2e}")
            prev_ux = ux.copy()

            saved_ux.append(np.sqrt(ux**2 + uy**2))
            ux_history.append(ux.copy())

            if residual < convergence_tol:
                print(f"✅ Converged at step {step}")
                break

    # Save output
    np.save(os.path.join(output_dir, "ux_final.npy"), ux)
    np.save(os.path.join(output_dir, "uy_final.npy"), uy)
    np.save(os.path.join(output_dir, "rho_final.npy"), rho)
    np.savez_compressed(os.path.join(output_dir, "ux_snapshots.npz"), *saved_ux)
    np.savez_compressed(os.path.join(output_dir, "ux_history.npz"), **{f"frame_{i}": f for i, f in enumerate(saved_ux)})

    with open(os.path.join(output_dir, "residuals.csv"), "w", newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["Step", "Residual"])
        for s, r in zip(steps, residuals):
            writer.writerow([s, r])

    print("✅ Simulation finished. Results saved in 'output/'")

if __name__ == "__main__":
    run_simulation()
