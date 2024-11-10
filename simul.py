import pyvista as pv
import numpy as np
import meshio
from sfepy.discrete import Problem
from sfepy.discrete.fem import FEDomain
from sfepy.discrete.fem.mesh import Mesh as SfePyMesh
from sfepy.discrete.common.fields import Field
from sfepy.discrete import Integral, FieldVariable
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.terms import Term
from sfepy.discrete.equations import Equations, Equation
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms.terms_surface import LinearTractionTerm

# Load and visualize the STL mesh using PyVista
file_path = "D:/malo/Documents/post prepa/info/design for am/devoir2/original.STL"
mesh = pv.read(file_path)

# Function to find indices based on coordinates with tolerance
def find_indices(mesh_points, target_coords, atol=5):
    indices = []
    for i, point in enumerate(mesh_points):
        if any(np.allclose(point, coord, atol=atol) for coord in target_coords):
            indices.append(i)
    return indices

# Define fixed and load points based on approximate locations
fixed_points_coords = [
    [154.16644287, 22.53479195, 52.67448807],
    [153.5009613, 48.94762039, 30.06476021],
    [23.97496986, 25.72593307, 49.6015358],
    [24.73397636, 63.79221725, 16.36836243]
]
load_points_coords = [
    [78.15579224, 26.76436424, 85.2099762],
    [100.38079071, 31.54273987, 83.92961884]
]

# Find fixed and load points' indices
fixed_points_indices = find_indices(mesh.points, fixed_points_coords, atol=5)
load_points_indices = find_indices(mesh.points, load_points_coords, atol=5)

# Convert the mesh to unstructured grid for further processing
mesh = mesh.cast_to_unstructured_grid()
vtk_file_path = "bracket_mesh.vtk"
mesh.save(vtk_file_path)

# Convert VTK to XDMF for SfePy compatibility
converted_mesh = meshio.read(vtk_file_path)
xdmf_file_path = "bracket_mesh.xdmf"
meshio.write(xdmf_file_path, converted_mesh, file_format="xdmf")

# Load mesh into SfePy
sfepy_mesh = SfePyMesh.from_file(xdmf_file_path)
domain = FEDomain("domain", sfepy_mesh)

# Define a region for the entire domain
region = domain.create_region("Omega", "all")

# Define Field and Problem
field = Field.from_args("displacement", np.float64, "vector", region, approx_order=1)
u = FieldVariable("u", "unknown", field)
v = FieldVariable("v", "test", field, primary_var_name="u")

# Material properties for steel
E, nu = 210e9, 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define problem formulation
integral = Integral("i", order=3)
elasticity_term = Term.new(
    "dw_lin_elastic_iso(m.lmbda, m.mu, v, u)",
    integral, region,
    m={"lmbda": lmbda, "mu": mu},
    v=v, u=u
)

# Convert the list of indices to a comma-separated string for SfePy compatibility
fixed_points_indices_str = ",".join(map(str, fixed_points_indices))
load_points_indices_str = ",".join(map(str, load_points_indices))

# Define regions for fixed points and load points using SfePy's syntax
fixed_region = domain.create_region(
    "FixedRegion",
    f"nodes by select_mesh_vertices({fixed_points_indices_str})",
    kind="facet"
)
load_region = domain.create_region(
    "LoadRegion",
    f"nodes by select_mesh_vertices({load_points_indices_str})",
    kind="facet"
)

# Define Essential boundary conditions (Dirichlet BC) for fixed points
ebcs = Conditions([
    EssentialBC("fixed", fixed_region, {"u.all": 0.0})
])

# Define load condition (Neumann BC) using SurfaceTractionTerm on the load region
load_value = np.array([0.0, 0.0, -1000.0])  # Example force vector
load_term = SurfaceTractionTerm(
    "dw_surface_ltr", integral, load_region, traction=load_value, v=v
)

# Combine terms into equations
equation = Equation("elasticity", elasticity_term + load_term)
equations = Equations([equation])

# Solver setup
ls = ScipyDirect({})
nls = Newton({}, lin_solver=ls)

# Define and solve the problem
pb = Problem("elasticity", equations=equations)
pb.time_update(ebcs=ebcs)
state = pb.solve()

# Visualize the displacement field using PyVista
displacement = state.get_state_parts()["u"]
mesh = pv.read(vtk_file_path)
mesh.point_data["displacement"] = displacement

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_arrows(mesh.points, displacement.reshape(-1, 3), mag=0.1, color="red")
plotter.add_axes()
plotter.show_grid()
plotter.show()
