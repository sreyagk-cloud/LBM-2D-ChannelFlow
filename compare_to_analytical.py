import numpy as np
import matplotlib.pyplot as plt

# Load LBM results
data = np.load('output/ux_history.npz')
frames = sorted([key for key in data.files if key.startswith('frame_')], key=lambda x: int(x.split('_')[1]))
ux_final = data[frames[-1]]  # Last time step

# Extract the centerline profile (u_x at mid-x across y)
mid_x = ux_final.shape[1] // 2
ux_centerline = ux_final[:, mid_x]

# Normalize LBM velocity profile
ux_norm = ux_centerline / np.max(ux_centerline)

# Create y-array normalized to domain height
Ny = ux_centerline.shape[0]
y = np.linspace(0, 1, Ny)

# Analytical parabolic profile: u(y) = 4 * u_max * y * (1 - y)
u_analytical = 4 * y * (1 - y)  # already normalized to u_max

# Plot comparison
plt.figure()
plt.plot(y, ux_norm, label='LBM (normalized)', linewidth=2)
plt.plot(y, u_analytical, '--', label='Analytical', linewidth=2)
plt.xlabel('y')
plt.ylabel('u / u_max')
plt.title('Velocity Profile Comparison at Final Time Step')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
