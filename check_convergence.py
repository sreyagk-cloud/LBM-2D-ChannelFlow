import numpy as np
import matplotlib.pyplot as plt

# Load all frames
data = np.load("output/ux_history.npz")
frame_keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))

# Load frames into an array
ux_history = np.array([data[k] for k in frame_keys])  # shape: (n_frames, ny, nx)

# Compute max norm of difference between successive frames
diffs = []
for i in range(1, len(ux_history)):
    diff = np.max(np.abs(ux_history[i] - ux_history[i - 1]))
    diffs.append(diff)

# Plot convergence trend
plt.figure(figsize=(8, 5))
plt.plot(diffs, label="Max difference between frames")
plt.yscale("log")
plt.xlabel("Frame index")
plt.ylabel("Max |Δu|")
plt.title("Convergence Check: Successive Frame Differences")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output/convergence_plot.png")
plt.show()
