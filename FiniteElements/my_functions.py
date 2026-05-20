import numpy as np
from dolfinx import mesh, fem # type: ignore

def create_quad_domain(comm, nx, ny, p0, p1, p2, p3, celltype=mesh.CellType.triangle):
    """
    Create a domain defined by four corner points.

    Args:
        comm (mpi4py.MPI.Intracomm): MPI communicator
        nx (int): number of cells in x-direction
        ny (int): number of cells in y-direction
        p0 (np.array): x and y coordinates of the bottom-left corner point
        p1 (np.array): x and y coordinates of the bottom-right corner point
        p2 (np.array): x and y coordinates of the top-right corner point
        p3 (np.array): x and y coordinates of the top-left corner point

    Returns:
        dolfinx.mesh: Domain defined by the corner points.
    """

    msh = mesh.create_unit_square(comm, nx, ny, cell_type=celltype)

    x = msh.geometry.x
    xi = x[:, 0]
    eta = x[:, 1]

    x[:, :2] = (
        np.outer((1-xi)*(1-eta), p0) +
        np.outer(xi*(1-eta), p1) +
        np.outer(xi*eta, p2) +
        np.outer((1-xi)*eta, p3)
    )
    return msh

def print_matrix_from_equation(Eq):
    X = fem.petsc.assemble_matrix(fem.form(Eq))
    X.assemble()
    X.convert("dense")
    C = X.getDenseArray()
    print(f"Matrix form: {C}")