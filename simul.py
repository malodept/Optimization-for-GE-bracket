import pyvista as pv
import numpy as np
import meshio
from sfepy.discrete import Problem
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.discrete.fem import Mesh
from sfepy.discrete.fem.domain import FEDomain
from sfepy.discrete.fem.mesh import Mesh as SfePyMesh
from sfepy.discrete.common.fields import Field
from sfepy.discrete import Integral, FieldVariable
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.terms import Term
from sfepy.discrete.equations import Equations, Equation
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

# Load the STL file
file_path = "D:/malo/Documents/post prepa/info/design for am/devoir2/original.STL"
mesh = pv.read(file_path)

# Visualize the original mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_axes()
plotter.show_grid()

# Callback function to print coordinates of picked points
def print_point_coords(picked_point, plotter):
    print("Picked point coordinates:", picked_point)

# Enable point picking with callback
plotter.enable_point_picking(callback=print_point_coords, show_message=True, use_picker=True)
plotter.show()

# Define the fixed points based on approximate locations
fixed_points_indices = []  # Indices for points in Interfaces 2-5 (to fix)
fixed_points_coords = [
    [154.16644287, 22.53479195, 52.67448807],  # Top left
    [153.5009613, 48.94762039, 30.06476021],   # Bottom left
    [23.97496986, 25.72593307, 49.6015358],    # Top right
    [24.73397636, 63.79221725, 16.36836243]    # Bottom right
]

# Find indices for fixed points based on coordinates
for i, point in enumerate(mesh.points):
    for coord in fixed_points_coords:
        if np.allclose(point, coord, atol=5):
            fixed_points_indices.append(i)

# Define points under load (Interface 1)
load_points_indices = []  # Indices for points on Interface 1
load_points_coords = [
    [78.15579224, 26.76436424, 85.2099762],   # Right
    [100.38079071, 31.54273987, 83.92961884]  # Left
]

# Find indices for load points based on coordinates
for i, point in enumerate(mesh.points):
    for coord in load_points_coords:
        if np.allclose(point, coord, atol=5):
            load_points_indices.append(i)

# Convert POLYDATA to UNSTRUCTURED_GRID for further processing
mesh = mesh.cast_to_unstructured_grid()

# Save the original mesh in VTK format for further use with SfePy
vtk_file_path = "bracket_mesh.vtk"
mesh.save(vtk_file_path)

# Convert and save as XDMF for SfePy compatibility
converted_mesh = meshio.read(vtk_file_path)
xdmf_file_path = "bracket_mesh.xdmf"
meshio.write(xdmf_file_path, converted_mesh, file_format="xdmf")

# Load mesh into SfePy
sfepy_mesh = SfePyMesh.from_file(xdmf_file_path)
domain = FEDomain("domain", sfepy_mesh)

# Define a region for the entire domain (volume elements)
region = domain.create_region("Omega", "all")

# Define Field and Problem
field = Field.from_args("displacement", np.float64, "vector", region, approx_order=1)
u = FieldVariable("u", "unknown", field)
v = FieldVariable("v", "test", field, primary_var_name="u")

# Material properties for steel
E, nu = 210e9, 0.3  # Young's modulus and Poisson's ratio
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define problem formulation
integral = Integral("i", order=3)
t1 = Term.new("dw_lin_elastic_iso(m.lmbda, m.mu, v, u)", integral, region, m={"lmbda": lmbda, "mu": mu}, v=v, u=u)

# Define load condition (Neumann BC)
load_value = np.array([0.0, 0.0, -1000.0])
load = Term.new("dw_surface_ltr", integral, region, load=load_value, primary_var_name="u")

# Define Essential boundary conditions (Dirichlet BC)
ebcs = Conditions([
    EssentialBC("fixed", region, {"u.all": 0.0})
])

# Combine the terms into equations
eq = Equation("elasticity", t1 - load)
eqs = Equations([eq])

# Solver setup
ls = ScipyDirect({})
nls = Newton({}, lin_solver=ls)

# Define the problem
pb = Problem("elasticity", equations=eqs)
pb.time_update(ebcs=ebcs)

# Apply a small load effect (for visualization)
for idx in load_points_indices:
    mesh.points[idx] += [0, 0, -5]  # Adjust this value for more/less deformation

# Visualize the simulated mesh with load effect
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="salmon", show_edges=True)
plotter.add_axes()
plotter.show_grid()
plotter.show()

# After solving the problem, visualize the displacement field using PyVista
mesh = pv.read("bracket_mesh.vtk")
displacement = u.evaluate_at_nodes()  # Evaluate the displacement field
mesh.point_data["displacement"] = displacement  # Assign to mesh for visualization

# Visualize the displacement field
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_arrows(mesh.points, displacement, mag=0.1, color="red")  # Adjust 'mag' for scale
plotter.add_axes()
plotter.show_grid()
plotter.show()
