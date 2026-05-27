import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from my_functions import *
from visualization_fct import *
import time
from nonlinear_snes_problem import NonlinearPDE_SNESProblem
import pickle


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

    return bc, inflow




def solve_with_Newton(nx, nz, P0, P1, P2, P3, T, delta_t = 7, save_tmp=False):
    from dolfinx.fem.petsc import create_matrix, create_vector

    domain = create_quad_domain(MPI.COMM_WORLD, nx, nz, P0, P1, P2, P3, celltype=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("CG", 1))
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    # Define constants
    delta_t = fem.Constant(domain, PETSc.ScalarType(delta_t))
    # Define constants (silt)
    alpha = fem.Constant(domain, PETSc.ScalarType(1.6))
    N = fem.Constant(domain, PETSc.ScalarType(1.37))
    theta_r = fem.Constant(domain, PETSc.ScalarType(0.034))
    theta_s = fem.Constant(domain, PETSc.ScalarType(0.46))
    Ks = fem.Constant(domain, PETSc.ScalarType(6.94e-7))

    # van Genuchten
    def S_e(h_w):
        return ufl.conditional(h_w < 0, (1 + (- alpha * h_w)**N)**((1 - N) / N), 1)
    def thetafct(Se):
        return theta_r + (theta_s - theta_r)*Se
    def k_rel(Se):
        m = 1 - 1/N
        return ufl.conditional(Se < 1 - 1e-7, ufl.sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2 , 1)

    bc, inflow = boundary_conditions(V, domain, v, P0, P1, P2, P3)

    t = 0.0 # start time [s]
    
    # Variational formulation
    h_w_old = fem.Function(V)
    h_w_old.name = "h_w_old"
    h_w_old.x.array[:] = -1e-2*np.ones_like(h_w_old.x.array) # initial condition close to saturation

    h_w_new = fem.Function(V)

    z = x[1]   

    
    # Create Newton solver
    snes = PETSc.SNES().create()

    b = create_vector(V)

    # Weak formulation
    F = (thetafct(S_e(h_w_new)) - thetafct(S_e(h_w_old))) / delta_t * v * ufl.dx
    F += ufl.dot(ufl.grad(v), Ks * k_rel(S_e(h_w_new)) * ufl.grad(z + h_w_new)) * ufl.dx
    F += inflow

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, h_w_new, bc)
    J = create_matrix(problem.a)

    # Create structure for saving temporary files
    if save_tmp:
        tmp = {
            "nx": nx,
            "nz": nz,
            "T_end": T,
            "h_w": [],
            "times": []
            }
        next_saving_time = 1
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
        snes.setTolerances(rtol=1e-4, atol=1e-10, max_it=50)
        ksp = snes.getKSP()
        ksp.setType("gmres") # iterative solver
        #ksp.setType("preonly")
        ksp.setTolerances(rtol=1e-4)
        ksp.setErrorIfNotConverged(True)
        #ksp.setMonitor(None)
        #ksp.getPC().setType("lu")
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.getPC().setHYPREType("boomeramg")
    
        sol_vec = h_w_new.x.petsc_vec.copy() # create solution vector
        sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        snes.solve(None, sol_vec) # solve, store solution in solution vector

        sol_vec.copy(h_w_new.x.petsc_vec)
        h_w_new.x.scatter_forward()
        
        converged = snes.getConvergedReason()
        num_iter = snes.getIterationNumber()
        
        # adaptive time stepping:
        if num_iter > 10 and float(delta_t.value) > 1e-1:
            delta_t.value *= 0.5
            continue
        if num_iter < 3 and float(delta_t.value) < 3600:
            delta_t.value *= 1.2
        assert converged > 0, f"Solver did not converge, got {converged}."
        print(
            f"Solver converged after {num_iter} iterations with converged reason {converged}. Time step is {delta_t.value:.2f} s at t={t/3600:.2f} hours."
        )

        # save temporary data
        if save_tmp and np.isclose(t/3600, next_saving_time):
            next_saving_time += 1
            tmp["h_w"].append(h_w_new.x.array)
            tmp["times"].append(t)


        t += float(delta_t.value)
        # update old solution
        h_w_old.x.array[:] = h_w_new.x.array

    # dump temporary data into pickle file
    if save_tmp:
        file_name = "./solutions/homogeneous_" + str(int(T/3600)) + "h_nx" + str(nx) + "_nz" + str(nz) + ".pkl"
        with open(file_name, "wb") as f:
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
delta_x = delta_z = 0.3
nx = int(6/delta_x)
nz = int(3/delta_x)
print(f"Resolution is dx = {delta_x} m, dz = {delta_z} m, giving nx = {nx}, nz = {nz}")

T = 48*60*60 # end time [s]
delta_t = 7 # time step [s]

t0 = time.time()
h_w = solve_with_Newton(nx, nz, P0, P1, P2, P3, T, delta_t, save_tmp=True)
elapsed = time.time() - t0
#np.save(f"./solutions/transient_{T/3600:.0f}h_nx{nx}_nz{nz}" , h_w.x.array)
print(f"Time needed for execution: {elapsed:.2f} s.")