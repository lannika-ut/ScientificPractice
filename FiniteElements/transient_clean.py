import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from my_functions import *

# Define geometry
P0 = np.array([0, 0])
P1 = np.array([6, -1])
P2 = np.array([6, 2])
P3 = np.array([0, 3])
slope = (P1[1]-P0[1])/(P1[0]-P0[0])

nx = nz = 10

# Generate mesh from corner points
domain = create_quad_domain(MPI.COMM_WORLD, nx, nz, P0, P1, P2, P3, celltype=mesh.CellType.quadrilateral)

# Initialize functionspace, test function, coordinates
V = fem.functionspace(domain, ("CG", 1))
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

#######################################################################
# Boundary conditions
#######################################################################
# Boundary condition locators
def on_dirichlet(x):
    return np.logical_and(np.isclose(x[0], P1[0]), x[1] <= 0)

def top(x):
    return np.isclose(x[1], slope*x[0]+P3[1])

# Dirichlet boundary
dofs_D = fem.locate_dofs_geometrical(V, on_dirichlet)
bc = fem.dirichletbc(PETSc.ScalarType(0), dofs_D, V)

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

#######################################################################
# Parametrization
#######################################################################
# Define constants (silt)
alpha = 1.6
N = 1.37
theta_r = 0.034
theta_s = 0.46
Ks = 6.94e-7

# van Genuchten
def S_e(h_c):
    return ufl.conditional(h_c < 0, (1 + (- alpha * h_c)**N)**((1 - N) / N), 1)
def theta(Se):
    return theta_r + (theta_s - theta_r)*Se
def k_rel(Se):
    m = 1 - 1/N
    return ufl.conditional(Se < 1 - 1e-7, ufl.sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2 , 1)

#######################################################################
# Time discretization
#######################################################################
t = 0.0 # start time [s]
T = 60*60 # end time [s]
delta_t = 7 # time step [s]


#######################################################################
# Start solving
#######################################################################

h_c_old = fem.Function(V)
h_c_old.name = "h_c_old"
h_c_old.x.array[:] = -1e-2*np.ones_like(h_c_old.x.array) # initial condition close to saturation

h_c_new = fem.Function(V)
h_tot = fem.Function(V)
z = x[1]

# Solver options
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-4,
    "snes_rtol": 1e-4,
    "snes_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

# Time loop
while t <= T:
    F = (theta(S_e(h_c_new)) - theta(S_e(h_c_old))) / delta_t * v * ufl.dx
    F += ufl.dot(ufl.grad(v), Ks * k_rel(S_e(h_c_new)) * ufl.grad(z + h_c_new)) * ufl.dx
    F += inflow
    problem = NonlinearProblem(
        F,
        h_c_new,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="richards",
    )
    h_c_new = problem.solve()
    converged = problem.solver.getConvergedReason()
    num_iter = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge, got {converged}."
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}."
    )
    h_c_old.x.array[:] = h_c_new.x.array
    t += delta_t