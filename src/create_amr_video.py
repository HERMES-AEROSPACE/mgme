import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['font.family'] = "serif"
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlabel(r'C_x', fontsize=18)
ax1.set_ylabel('f', fontsize=18)
ax1.tick_params(axis='both', labelsize=14)

with h5py.File("amr_showcase_data.h5", "r") as hf:
    cx_vec = hf["cx_vec"][:]
    frames = sorted(hf.keys() - {"cx_vec"})  # all frame_XXXX groups

    data = []
    for frame in frames:
        grp = hf[frame]
        data.append({
            "t":       grp.attrs["t"],
            "h_dist1": grp.attrs["h_dist1"],
            "h_dist2": grp.attrs["h_dist2"],
            "refined": grp.attrs["refined"],
            "ft":      grp["ft"][:],
            "f":       grp["f"][:],
            "f1":      grp["f1"][:],
            "f2":      grp["f2"][:],
        })

def animate(i):
    ax1.clear()
    ax1.set_xlabel(r'C_x', fontsize=18)
    ax1.set_ylabel('f', fontsize=18)
    ax1.tick_params(axis='both', labelsize=14)

    fd = data[i]
    ax1.plot(cx_vec, fd["ft"], color='black')

    if fd["refined"]:
        ax1.plot(cx_vec[0:61],   fd["f1"], color='indigo')
        ax1.plot(cx_vec[60:121], fd["f2"], color='firebrick')
        ax1.legend(['test f', 'group 1', 'group 2'], fontsize=14, loc='upper right')
    else:
        ax1.plot(cx_vec, fd["f"], color='firebrick')
        ax1.legend(['test f', 'group 1'], fontsize=14, loc='upper right')

    ax1.set_ylim(-0.05, 0.6)
    ax1.set_title(f"t={fd['t']}  H1={fd['h_dist1']:.4f}  H2={fd['h_dist2']:.4f}")

anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=100)
anim.save('amr_showcase.gif', writer='pillow', fps=10)
# or just: plt.show() to view interactively