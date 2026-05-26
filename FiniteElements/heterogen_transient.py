import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from my_functions import *
from visualization_fct import *

# Define geometry
P0 = np.array([0, 0])
P1 = np.array([6, -1])
P2 = np.array([6, 2])
P3 = np.array([0, 3])
slope = (P1[1]-P0[1])/(P1[0]-P0[0])

nx = nz = 20 

# Generate mesh from corner points
domain = create_quad_domain(MPI.COMM_WORLD, nx, nz, P0, P1, P2, P3, celltype=mesh.CellType.quadrilateral)

# Initialize functionspace, test function, coordinates
Q = fem.functionspace(domain, ("DG", 0)) # material parameters
V = fem.functionspace(domain, ("CG", 1)) # pressure head
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)


# -------------------------------------------------
# DG0 parameter fields
# -------------------------------------------------

Ks      = fem.Function(Q)
alpha   = fem.Function(Q)
n_vg    = fem.Function(Q)
theta_r = fem.Function(Q)
theta_s = fem.Function(Q)

# -------------------------------------------------
# Material assignment
# -------------------------------------------------

# Define parameters for each layer (layer 1, top: sand, layer 2, bottom: silt)
layer_params = {
    1: {"name": "loam", "alpha": 3.6, "N": 1.56, "theta_r": 0.078, "theta_s": 0.43, "Ks": 2.89e-6, "locator": lambda x: False},
    2: {"name": "sand", "alpha": 14.5, "N": 2.68, "theta_r": 0.045, "theta_s": 0.43, "Ks": 8.25e-5, "locator": lambda x: x[1] >= slope*x[0] + P3[1]/2 - 1e-14},
    3: {"name": "silt", "alpha": 1.6, "N": 1.37, "theta_r": 0.034, "theta_s": 0.46, "Ks": 6.94e-7, "locator": lambda x: x[1] < slope*x[0] + P3[1]/2 - 1e-14},
}

tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
cells = np.arange(num_cells, dtype=np.int32)

midpoints = mesh.compute_midpoints(domain, tdim, cells)

Ks_vals      = np.zeros(num_cells)
alpha_vals   = np.zeros(num_cells)
n_vals       = np.zeros(num_cells)
theta_r_vals = np.zeros(num_cells)
theta_s_vals = np.zeros(num_cells)

for c, x in enumerate(midpoints):
    # Check in which layer the midpoint is and assign the corresponding parameters
    for key, value in layer_params.items():
        if value["locator"](x):
            Ks_vals[c] = value["Ks"]
            alpha_vals[c] = value["alpha"]
            n_vals[c] = value["N"]
            theta_r_vals[c] = value["theta_r"]
            theta_s_vals[c] = value["theta_s"]

# -------------------------------------------------
# Store into DG0 functions
# -------------------------------------------------

Ks.x.array[:] = Ks_vals
alpha.x.array[:] = alpha_vals
n_vg.x.array[:] = n_vals
theta_r.x.array[:] = theta_r_vals
theta_s.x.array[:] = theta_s_vals

Ks.x.scatter_forward()
alpha.x.scatter_forward()
n_vg.x.scatter_forward()
theta_r.x.scatter_forward()
theta_s.x.scatter_forward()

# -------------------------------------------------
# Define parametrizations
# -------------------------------------------------

# van Genuchten
def S_e(h_w, alpha, N):
    return ufl.conditional(h_w < 0, (1 + (- alpha * h_w)**N)**((1 - N) / N), 1)
def theta(Se, theta_r, theta_s):
    return theta_r + (theta_s - theta_r)*Se
def k_rel(Se, N):
    m = 1 - 1/N
    return ufl.conditional(Se < 1 - 1e-7, ufl.sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2 , 1)

# Boundary condition locators
def on_dirichlet(x):
    return np.logical_and(np.isclose(x[0], P1[0]), x[1] <= 0)

def top(x):
    return np.isclose(x[1], slope*x[0]+P3[1])

# Dirichlet boundary
def dirichlet(x):
    return -x[1]
dofs_D = fem.locate_dofs_geometrical(V, on_dirichlet)
u_D = fem.Function(V)
u_D.interpolate(dirichlet)
bc = fem.dirichletbc(u_D, dofs_D)

# Inflow (Neumann) boundary
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology) 
facets_on_top = mesh.locate_entities(domain, fdim, top)
facets_not_top = np.setdiff1d(boundary_facets, facets_on_top)
# Mark facets belonging to inflow boundary with 1
facet_markers = np.zeros_like(boundary_facets)
facet_markers[:len(facets_on_top)] = 1
facet_indices = np.hstack([facets_on_top, facets_not_top])
sorted_facets = np.argsort(facet_indices)
# Create meshtags for facets (marker = 1 for inflow boundary)
facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]) 
# Create custom integration measure for inflow boundary
ds = ufl.Measure("ds", domain, subdomain_data=facet_tags)
# Recharge
c_in = fem.Constant(domain, PETSc.ScalarType(2e-9))
inflow = - v * c_in * ds(1)

# Time discretization
t = 0.0 # start time [s]
T = 24*60*60 # end time [s]
delta_t = 7 # time step [s]

# Variational formulation
h_w_old = fem.Function(V)
h_w_old.name = "h_w_old"
h_w_old.x.array[:] = -1e-2*np.ones_like(h_w_old.x.array) # initial condition close to saturation

h_w_new = fem.Function(V)
h_w_new.x.array[:] = h_w_old.x.array

z = x[1]

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-10,
    "snes_rtol": 1e-4,
    "snes_monitor_cancel": None,
    "snes_maxit": 20,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "ksp_monitor_cancel": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

while t <= T:
    t += delta_t

    # Initial guess for Newton
    h_w_new.x.array[:] = h_w_old.x.array

    # Weak formulation
    F = (theta(S_e(h_w_new, alpha, n_vg), theta_r, theta_s) - theta(S_e(h_w_old, alpha, n_vg), theta_r, theta_s)) / delta_t * v * ufl.dx
    F += ufl.dot(ufl.grad(v), Ks * k_rel(S_e(h_w_new, alpha, n_vg), n_vg) * ufl.grad(z + h_w_new)) * ufl.dx
    F += inflow
    problem = NonlinearProblem(
        F,
        h_w_new,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="richards",
    )
    h_w_new = problem.solve()
    converged = problem.solver.getConvergedReason()
    num_iter = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge, got {converged}."
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}. Time step is {delta_t} s at t={(t-delta_t)/3600:.2f} hours."
    )

    # adaptive time stepping:
    if num_iter > 10 and delta_t > 1e-1:
        t += - delta_t
        delta_t *= 0.5
        h_w_new.x.array[:] = h_w_old.x.array
    if num_iter < 3:
        delta_t *= 1.2

    # update old solution
    h_w_old.x.array[:] = h_w_new.x.array
    