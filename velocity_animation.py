import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Load ux history
data = np.load("output/ux_history.npz")
ux_frames = [data[f'frame_{i}'] for i in range(len(data.files))]

# Set up figure
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(ux_frames[0], origin='lower', cmap='viridis', interpolation='nearest')
fig.colorbar(cax, ax=ax, label='Ux velocity')

ax.set_title('2D Channel Flow: Ux Velocity')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Update function
def update(frame):
    cax.set_data(ux_frames[frame])
    ax.set_title(f'Frame {frame} | Max: {ux_frames[frame].max():.5f}')
    return [cax]

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(ux_frames), interval=200, blit=True)

# Save animation
output_path = "output/ux_animation.mp4"
os.makedirs("output", exist_ok=True)
ani.save(output_path, fps=5, dpi=150)

print(f"Animation saved to: {output_path}")
plt.close()
