import numpy as np
import matplotlib.pyplot as plt

# Load data from the correct folder
ux = np.load('output/ux_final.npy')  # shape: (nx, ny)

# Assuming the midline is horizontal (along the center of the channel height)
mid_y = ux.shape[1] // 2
midline = ux[:, mid_y]

# Plotting
x = np.arange(len(midline))
plt.plot(x, midline, label='Numerical')

# Add analytical solution for comparison, if available
# e.g., u_analytical = (4 * umax / H**2) * y * (H - y)
# plt.plot(x, u_analytical, label='Analytical')

plt.title("Midchannel Velocity Profile")
plt.xlabel("X")
plt.ylabel("Velocity (u)")
plt.legend()
plt.grid(True)
plt.show()
