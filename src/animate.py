import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .moment_utils import invert
from .config_1d import PHYS_SPACE, CONSTANTS, FREESTREAM_PARAMS
from scipy.interpolate import interp1d


dsmc = np.loadtxt('src/dsmc.txt')
dsmcT = np.loadtxt('src/dsmcT.txt')
alsmeyer_205 = np.loadtxt('src/alsmeyer_205.txt')
plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# ax.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='black')
# ax.plot(1 - dsmcT[:, 0], dsmcT[:, 1], '--', color='red')
line1, = ax.plot([], [], color='black', label='Density')
line2, = ax.plot([], [], color='red', label='Temperature')
line3, = ax.plot([], [], '--', color='black', label='Velocity')
lines = [line1, line2]

ax.set_xlabel('Scaled Location', fontsize=18)
ax.set_ylabel('Normalized Property', fontsize=18)
ax.legend(loc='upper left', fontsize=14)

R = CONSTANTS['R']
m = CONSTANTS['m']
T1 = FREESTREAM_PARAMS['T1']
P1 = FREESTREAM_PARAMS['P1']
d_ref = CONSTANTS['d']
gamma = CONSTANTS['gamma']
rho1    = P1 / (R * T1)
n_ref = P1/(R * T1) * 1/m
sigma_ref = np.pi * d_ref**2
lam_ref = 1/(n_ref * sigma_ref)

mu_argon_300K = 2.2948e-5   # Pa.s
lam_alsmeyer  = 16/5 * mu_argon_300K / (rho1 * np.sqrt(2 * np.pi * R * T1))
x = np.linspace(*PHYS_SPACE['xj_range'], PHYS_SPACE['num_xj'])
x_scale = x * (lam_ref / lam_alsmeyer)

ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-0.05, 1.15)
ax.tick_params(axis='both', labelsize=14)
plt.locator_params(axis='both', nbins=10)
ax.minorticks_on()
ax.grid()
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

    interp    = interp1d(n_scale, x_scale)
    x_center  = interp(0.5)

    line1.set_data(x_scale - x_center, n_scale)
    line2.set_data(x_scale - x_center, temperature_scale)
    line3.set_data(alsmeyer_205[:, 0], alsmeyer_205[:, 1])

    grad = np.max(np.abs(np.gradient(n_scale, x_scale)))

    # p = 52
    # negx_n = np.sum(data[p, 0:4], axis=0)[0]
    # posx_n = np.sum(data[p, 4:], axis=0)[0]
    # negx_u = np.sum(data[p, 0:4], axis=0)[1]
    # posx_u = np.sum(data[p, 4:], axis=0)[1]
    # negx_e = np.sum(data[p, 0:4], axis=0)[4]
    # posx_e = np.sum(data[p, 4:], axis=0)[4]
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
    t = i * 0.032
    title.set_text(f't = {t:.3f}, rst = {grad:.4f}')

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

anim = FuncAnimation(fig, update, init_func=init, frames=1080, blit=False, interval=50)
anim.save('plots/evo.gif')
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()