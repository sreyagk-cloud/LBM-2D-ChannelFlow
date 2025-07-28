import numpy as np
import matplotlib.pyplot as plt
import os

# Load final velocity fields
output_dir = "output"
ux = np.load(os.path.join(output_dir, 'ux_final.npy'))
uy = np.load(os.path.join(output_dir, 'uy_final.npy'))

# Compute velocity magnitude
u_mag = np.sqrt(ux**2 + uy**2)

# Plotting
plt.figure(figsize=(8, 4))
contour = plt.contourf(u_mag.T, levels=50, cmap='jet')  # Transpose to match x-y orientation
plt.colorbar(contour, label='Velocity Magnitude')
plt.title('Steady-State Velocity Contour')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("velocity_contour.png", dpi=300)  # Save as image
plt.show()
