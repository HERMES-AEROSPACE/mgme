import cvxpy as cp
import numpy as np
from scipy.stats import qmc
import cvxpy as cp
from itertools import product
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu


# Group definition.
ci_cx = -1.0
cf_cx = 0.0
ci_cy = -3.0
cf_cy = 0.0 
ci_cz = -3.0
cf_cz = 0.0

cx_vec = np.linspace(ci_cx, cf_cx, 30)
cy_vec = np.linspace(ci_cy, cf_cy, 30)
cz_vec = np.linspace(ci_cz, cf_cz, 30)

[cx, cy, cz] = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

# Create samples.
volume = (cf_cx - ci_cx) * \
        (cf_cy - ci_cy) * \
        (cf_cz - ci_cz)
num_samples = int(np.ceil(10 * volume))

sampler = qmc.LatinHypercube(d=3, seed=6767)
sample = qmc.scale(sampler.random(n=int(num_samples)), [ci_cx, ci_cy, ci_cz], [cf_cx, cf_cy, cf_cz])

x_sample = sample[:, 0]
y_sample = sample[:, 1]
z_sample = sample[:, 2]

# Create meshgrid of A, beta, wx, wy, and wz to represent all sorts of distributions.
A = np.logspace(-3, -2, 4)
b = np.logspace(-6, -4, 3)
wx = np.linspace(ci_cx, cf_cx, 5)
wy = np.linspace(ci_cy, cf_cy, 5)
wz = np.linspace(ci_cz, cf_cz, 5)

[A_mesh, b_mesh, wx_mesh, wy_mesh, wz_mesh] = np.meshgrid(A, b, wx, wy, wz, indexing='ij')

param_combinations = list(product(A, b, wx, wy, wz))
total_runs = len(param_combinations)
print(f"Total optimization runs required: {total_runs}")

# CVXPY parameters.
param_n = cp.Parameter(nonneg=True)       # Density
param_u = cp.Parameter(3)                 # Velocity vector
param_E = cp.Parameter(nonneg=True)       # Energy

x = cp.Variable(shape=num_samples, nonneg=True)
obj = cp.Maximize(cp.sum(cp.entr(x)))

constraints = [
    cp.sum(x) == param_n,
    cp.sum(cp.multiply(x_sample, x)) == param_u[0],
    cp.sum(cp.multiply(y_sample, x)) == param_u[1],
    cp.sum(cp.multiply(z_sample, x)) == param_u[2],
    cp.sum(cp.multiply(x_sample**2 + y_sample**2 + z_sample**2, x)) == param_E
]

prob = cp.Problem(obj, constraints)

# Loop.
data_rows = [] # List to store dicts

for i, params in enumerate(param_combinations):
    A_val, b_val, wx_val, wy_val, wz_val = params

    f = A_val * np.exp(-b_val * ((cx - wx_val)**2 + (cy - wy_val)**2 + (cz - wz_val)**2))

    mu = calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec)

    param_n.value = mu[0]
    param_u.value = np.array([mu[1], mu[2], mu[3]])
    param_E.value = mu[4]

    try:
        # warm_start=True can help if changes between steps are small
        prob.solve(warm_start=True) 
        
        if prob.status == 'optimal':
            status_code = 1 # Success
            opt_val = prob.value
        else:
            status_code = 0.5 # some error
            opt_val = -1
    except Exception as e:
        status_code = 0 # crash
        opt_val = -1

    # data_rows.append({
    #     'Amplitude': A_val,
    #     'Beta': np.log10(b_val), # Log scale is often better for plotting beta
    #     'Wx': wx_val,
    #     'Wy': wy_val,
    #     'Wz': wz_val,
    #     'Status': status_code,   # 1=Good, 0.5=Infeasible, 0=Crash/Empty
    #     'Obj_Value': opt_val
    # })

    data_rows.append({
        'Density': mu[0],
        'x-momentum': mu[1], # Log scale is often better for plotting beta
        'y-momentum': mu[2],
        'z-momentum': mu[3],
        'Energy': mu[4],
        'Status': status_code,   # 1=Good, 0.5=Infeasible, 0=Crash/Empty
        'Obj_Value': opt_val
    })

    if i % 100 == 0:
        print(f"Processed {i}/{total_runs} runs...")


df = pd.DataFrame(data_rows)
print(df.head())
cols_to_plot = {
    "Density": "n",
    "Energy": "rho*e",
    "x-momentum": "rho*ux", 
    "y-momentum": "rho*uy", 
    "z-momentum": "rho*uz",
    "Status": "Success (1.0)",
    'Obj_Value': 'objective'
}

# Create the dimensions list dynamically
dims = [
    dict(label=label, values=df[col_name]) 
    for col_name, label in cols_to_plot.items()
]

fig = go.Figure(data=
    go.Parcoords(
        # 1. Setup the Lines (Coloring)
        line = dict(
            color = df['Status'],
            colorscale = px.colors.diverging.Tealrose,
            cmin = 0,
            cmax = 1,
            showscale = True,
            colorbar = dict(title="Status")
        ),
        
        # 2. Setup the Axes (Dimensions)
        dimensions = dims
    )
)

fig.show()
