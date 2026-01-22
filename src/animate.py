import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .moment_utils import invert
from .config import PHYS_SPACE


dsmc = np.loadtxt('src/dsmc.txt')
dsmcT = np.loadtxt('src/dsmcT.txt')
plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# ax.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='black')
# ax.plot(1 - dsmcT[:, 0], dsmcT[:, 1], '--', color='red')
line1, = ax.plot([], [], color='black', label='Density')
line2, = ax.plot([], [], color='red', label='Temperature')
line3, = ax.plot([], [], color='blue', label='Velocity')
lines = [line1, line2]

ax.set_xlabel('Scaled Location', fontsize=18)
ax.set_ylabel('Normalized Property', fontsize=18)
ax.legend(loc='upper left', fontsize=14)

x = np.linspace(*PHYS_SPACE['xj_range'], PHYS_SPACE['num_xj'])
x_scale = (x - x.min()) / (x.max() - x.min())

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.05, 1.15)
ax.tick_params(axis='both', labelsize=14)
plt.locator_params(axis='both', nbins=10)
ax.minorticks_on()
title = ax.set_title('', fontsize=18)

anim_running = True

def init():
    title.set_text('')
    for line in lines:
        line.set_data([], [])
        
    return lines, title

def update(i):
    t = 0
    data = np.load('simulation_data/U{}.npy'.format(i + 0))
    n = np.sum(data, axis=1)[:, 0]
    u = np.sum(data, axis=1)[:, 1]
    e = np.sum(data, axis=1)[:, 4]
    temperature = 2/3 * ((e / n) - (u / n)**2)
    vel = u / n
    n_scale = (n - n[0]) / (n[-1] - n[0])
    vel_scale = (vel - vel[-1]) / (vel[0] - vel[-1])
    temperature_scale = (temperature - temperature[0]) / (temperature[-1] - temperature[0])
    shock_thick = (np.max(n_scale) - np.min(n_scale)) / np.max(np.abs(np.gradient(n_scale, x_scale)))
    thick = 1.098/shock_thick

    line1.set_data(x_scale, n_scale)
    line2.set_data(x_scale, temperature_scale)
    line3.set_data(x_scale, vel_scale)

    p = 52
    negx_n = np.sum(data[p, 0:4], axis=0)[0]
    posx_n = np.sum(data[p, 4:], axis=0)[0]
    negx_u = np.sum(data[p, 0:4], axis=0)[1]
    posx_u = np.sum(data[p, 4:], axis=0)[1]
    negx_e = np.sum(data[p, 0:4], axis=0)[4]
    posx_e = np.sum(data[p, 4:], axis=0)[4]
    # cx_vec, cy_vec, cz_vec = np.linspace(-5, 5.5, 106), np.linspace(-5, 5.5, 106), np.linspace(-5, 5.5, 106)
    # cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    # point = (x[p] - x.min()) / (x.max() - x.min())

    # A, b, wx, _, _ = invert([negx_n, negx_u, 0.0, 0.0, negx_e], [1.0, 0.0, 0.0, 0.0], {'ci_cx': -5, 'cf_cx': 1.2, 'ci_cy': -5, 'cf_cy': 5.5, 'ci_cz': -5, 'cf_cz': 5.5})
    # negx_f = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

    # A, b, wx, _, _ = invert([posx_n, posx_u, 0.0, 0.0, posx_e], [1.0, 0.0, 0.0, 0.0], {'ci_cx': 1.2, 'cf_cx': 5.5, 'ci_cy': -5, 'cf_cy': 5.5, 'ci_cz': -5, 'cf_cz': 5.5})
    # posx_f = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

    # line1.set_data(cx_vec[0:63], negx_f[0:63])
    # line2.set_data(cx_vec[62:], posx_f[62:])
    # if i > 115169
    #     title.set_text(f't = {t:.2f}')
    # else:
    t = i * 0.029
    title.set_text(f't = {t:.3f}, $u1$ = {posx_u/posx_n:.3f}, $u0$ = {negx_u/negx_n:.3f}')

    return lines, title

def on_key(event):
    global anim_running
    if event.key == ' ':  # Space bar
        if anim_running:
            anim.pause()
            anim_running = False
        else:
            anim.resume()
            anim_running = True

anim = FuncAnimation(fig, update, init_func=init, frames=130, blit=False, interval=50)
# anim.save('simulation_data/evo.mp4')
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()