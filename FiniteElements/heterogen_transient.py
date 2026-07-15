import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from my_functions import *
from visualization_fct import *
from nonlinear_snes_problem import NonlinearPDE_SNESProblem
import pickle
import time

# Boundary conditions
def boundary_conditions(V, domain, v, P0, P1, P2, P3):
    slope = (P1[1]-P0[1])/(P1[0]-P0[0])
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

    # Test constant head on top
    dofs_D_top = fem.locate_dofs_geometrical(V, top)
    u_D_top = fem.Function(V)
    u_D_top.x.array[:] = 0.1
    bc_top = fem.dirichletbc(u_D_top, dofs_D_top)

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

    return bc, bc_top

def solve_with_Newton(nx, nz, P0, P1, P2, P3, T, layer_params, delta_t = 7, save_tmp=False, filename=None):
    from dolfinx.fem.petsc import create_matrix, create_vector

    # Create FE framework
    domain = create_quad_domain(MPI.COMM_WORLD, nx, nz, P0, P1, P2, P3, celltype=mesh.CellType.quadrilateral)
    Q = fem.functionspace(domain, ("DG", 0)) # material parameters
    V = fem.functionspace(domain, ("CG", 1)) # pressure head
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Define delta_t as fem.Constant
    delta_t = fem.Constant(domain, PETSc.ScalarType(delta_t))

    # Assign material properties
    param_fct = assign_material(domain, Q, layer_params)
    Ks = param_fct["Ks"]
    alpha = param_fct["alpha"]
    N = param_fct["N"]
    theta_r = param_fct["theta_r"]
    theta_s = param_fct["theta_s"]

    # Get boundary conditions
    bc, bc_top = boundary_conditions(V, domain, v, P0, P1, P2, P3)

    t = 0.0 # start time [s]
    
    # Create Newton solver
    snes = PETSc.SNES().create()

    # Variational formulation
    h_w_old = fem.Function(V)
    h_w_old.name = "h_w_old"
    h_w_old.x.array[:] = -0.17*np.ones_like(h_w_old.x.array) # initial condition close to saturation

    h_w_new = fem.Function(V)

    z = x[1]

    # Weak formulation
    F = (theta(S_e(h_w_new, alpha, N), theta_r, theta_s) - theta(S_e(h_w_old, alpha, N), theta_r, theta_s)) / delta_t * v * ufl.dx
    F += ufl.dot(ufl.grad(v), Ks * k_rel(S_e(h_w_new, alpha, N), N) * ufl.grad(z + h_w_new)) * ufl.dx
    #F += inflow

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, h_w_new, [bc_top, bc])
    b = create_vector(V)
    J = create_matrix(problem.a)

    # Create structure for saving temporary files
    if save_tmp:
        tmp = {
            "nx": nx,
            "nz": nz,
            "T_end": T,
            "alpha": alpha.x.array,
            "N": N.x.array,
            "Ks": Ks.x.array,
            "theta_r": theta_r.x.array,
            "theta_s": theta_s.x.array,
            "h_w": [],
            "times": []
            }
        next_saving_time = 3600
        tmp["h_w"].append(h_w_old.x.array)
        tmp["times"].append(t)

    # Time loop
    while t <= T:

        # Initial guess for Newton
        h_w_new.x.array[:] = h_w_old.x.array
        
        snes.setFunction(problem.F, b) # assemble residual
        snes.setJacobian(problem.J, J) # assemble Jacobian

        # Set options
        snes.setType("newtonls")
        snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BT)
        snes.setTolerances(rtol=1e-4, atol=1e-11, max_it=20)
        ksp = snes.getKSP()
        ksp.setType("gmres") # iterative solver
        ksp.setTolerances(rtol=1e-4)
        ksp.setErrorIfNotConverged(True)
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.getPC().setHYPREType("boomeramg")
    
        sol_vec = h_w_new.x.petsc_vec.copy() # create solution vector
        sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        snes.solve(None, sol_vec) # solve, store solution in solution vector

        sol_vec.copy(h_w_new.x.petsc_vec) # copy solution into h_w_new
        h_w_new.x.scatter_forward()
        
        converged = snes.getConvergedReason()
        num_iter = snes.getIterationNumber()
        
        # adaptive time stepping:
        if num_iter > 10 and float(delta_t.value) > 1e-2:
            delta_t.value = max(0.5*float(delta_t.value), 1e-1)
            continue
        if num_iter < 3 and float(delta_t.value) < 3600:
            delta_t.value = min(float(delta_t.value)*1.2, 3600)
        assert converged > 0, f"Solver did not converge, got {converged}."
        print(
            f"Solver converged after {num_iter} iterations with converged reason {converged}. Time step is {delta_t.value:.2f} s at t={t/3600:.2f} hours."
        )

        # save temporary data
        if save_tmp and t >= next_saving_time:
            next_saving_time += 3600
            tmp["h_w"].append(h_w_new.x.array.copy())
            tmp["times"].append(t)

        # update time
        t += float(delta_t.value)
        # update old solution
        h_w_old.x.array[:] = h_w_new.x.array
    
    # save final data
    if save_tmp:
        tmp["h_w"].append(h_w_new.x.array.copy())
        tmp["times"].append(t)
    # dump temporary data into pickle file
    if save_tmp:
        if filename is None: # set default filename
            filename = "./solutions/heterogeneous_" + str(int(T/3600)) + "h_nx" + str(nx) + "_nz" + str(nz) + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(tmp, f)
    snes.destroy()
    b.destroy()
    J.destroy()

    return h_w_new
    


# Define geometry
P0 = np.array([0, 0])
P1 = np.array([6, -1])
P2 = np.array([6, 2])
P3 = np.array([0, 3])
slope = (P1[1]-P0[1])/(P1[0]-P0[0])
delta_x = delta_z = 0.1
nx = int(6/delta_x)
nz = int(3/delta_x)
print(f"Resolution is dx = {delta_x} m, dz = {delta_z} m, giving nx = {nx}, nz = {nz}")



# Define parameters for each layer (layer 1, top: sand, layer 2, bottom: silt)
layer_params = {
    1: {"name": "loam", "alpha": 3.6, "N": 1.56, "theta_r": 0.078, "theta_s": 0.43, "Ks": 2.89e-6, "locator": lambda x: False},
    2: {"name": "sand", "alpha": 14.5, "N": 2.68, "theta_r": 0.045, "theta_s": 0.43, "Ks": 8.25e-5, "locator": lambda x: x[1] >= slope*x[0] + P3[1]/2 - 1e-14},
    3: {"name": "silt", "alpha": 1.6, "N": 1.37, "theta_r": 0.034, "theta_s": 0.46, "Ks": 6.94e-7, "locator": lambda x: x[1] < slope*x[0] + P3[1]/2 - 1e-14},
}
layer_params = {1: {"name": "snow", "alpha": 4.99, "N": 14.56, "theta_r": 0.02, "theta_s": 0.9*0.468, "Ks": 6.859e-04, "locator": lambda x: True}}

T = 4*60*60
t0 = time.time()
h_w = solve_with_Newton(nx, nz, P0, P1, P2, P3, T, layer_params, save_tmp=True, filename="./Masterarbeit/solutions/iso_test_Dirichlet.pkl")
elapsed = time.time() - t0
print(f"Time needed for execution: {elapsed:.2f} s.")
