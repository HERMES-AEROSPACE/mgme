"""Velocity-space grid helpers shared by the 0-D and 1-D drivers."""
import numpy as np


def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz
