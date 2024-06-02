from dolfin import *
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

def complex_mesh_2d(r: int = 100) -> Tuple[Mesh, csr_matrix, np.ndarray]:
    """
    Generate a complex 2D mesh, assemble the system matrix, and return the mesh, 
    system matrix in CSR format, and the right-hand side vector.

    Args:
        r (int): Resolution of the mesh.

    Returns:
        tuple: Contains the following elements:
            - mesh (Mesh): The generated mesh.
            - A_sparray (csr_matrix): The system matrix in CSR format.
            - rhs (np.ndarray): The right-hand side vector.
    """
    # Create circles as Circle(Center, Radius)
    circle1 = Circle(Point(0, 0), 5)
    circle2 = Circle(Point(-1, 0), 1)

    domain = circle1 - circle2
    mesh = generate_mesh(domain, r)

    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = Expression("x[0] - x[1]", degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)

    # Define parameters for convection-diffusion
    epsilon = Constant(1.0)  # Diffusion coefficient
    b = Constant((1.0, 0.0))  # Convection velocity vector
    c = Constant(0.0)         # Reaction coefficient

    a = (dot(grad(u), epsilon * grad(v)) + dot(b, grad(u)) * v + c * u * v) * dx
    L = f * v * dx

    A, b = assemble_system(a, L, bc)
    A_mat = as_backend_type(A).mat()

    A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)

    rhs = b.get_local()

    return mesh, A_sparray, rhs
