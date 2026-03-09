import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv # type: ignore
from dolfinx import fem, io, mesh, plot, geometry # type: ignore


def plot_mesh(V):
    pv.set_jupyter_backend("html")
    cells, types, x = plot.vtk_mesh(V) # convert mesh to vtk data which pyvista can read
    grid = pv.UnstructuredGrid(cells, types, x)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        print("pyvista needs to be used in the default setting of pyvista.OFF_SCREEN=False.")


def evaluate_fct(domain, points, fcts):
    """
    This is a wrapper function to evaluate multiple dolfinx.fem.function.Function at given points. Explanations to what is going on can either be found in the script FEM_DeflectionOfAMembrane.ipynb or online: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html.

    Args:
        domain (dolfinx.mesh.Mesh): Mesh containing the topology (i.e. the cells).
        points (np.ndarray): Points at which the functions should be evaluated. Shape should be (3, num_points) with x-coordinates in the first, y-coordinates in the second and z-coordinates in the third dimension.
        fcts (list of dolfinx.fem.function.Function): Functions that need to be evaluated. The functions should be a linear combination of the basis functions on the domain, either created by interpolating an expression on a functionspace or by a finite element algorithm.

    Returns:
        list: points_on_proc (np.ndarray), fcts_valuess (list of np.ndarray): points where the functions are evaluated that are on the current processor and list of evaluated points for each function in fcts.
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, 
                                                       cell_candidates, 
                                                       points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate functions
    fcts_values = []
    for f in fcts:
        fcts_values.append(f.eval(points_on_proc, cells))
    return points_on_proc, fcts_values
        

def plotScalarFunction(V, u, warped=False, name = "u"):
    """
    Plot a dolfinx.function on its grid.

    Args:
        V (dolfinx.fem.function.FunctionSpace): Functionspace of the function, containing the grid.
        u (dolfinx.fem.function.Function): Scalar function that should be plotted.
        warped (bool, optional): If the plot should be warped to see changes in function values in 3D or not. Defaults to False.
        name (String): Name of the function to add to colour bar.
    """
    pv.set_jupyter_backend("html")
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
    grid.point_data[name] = u.x.array.real
    grid.set_active_scalars(name)
    plotter = pv.Plotter()
    if warped:
        warp = grid.warp_by_scalar()
        plotter.add_mesh(warp, show_edges = True, show_scalar_bar = True)
    else:
        plotter.add_mesh(grid, show_edges = True)
    plotter.view_xy()
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        print("pyvista needs to be used in the default setting of pyvista.OFF_SCREEN=False.")