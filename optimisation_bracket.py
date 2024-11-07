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
from sfepy.discrete.conditions import Conditions, EssentialBC, LinearCombinationBC
from sfepy.terms import Term
from sfepy.discrete.equations import Equations, Equation
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton



# Load the STL file
file_path = "D:/malo/Documents/post prepa/info/design for am/devoir2/original.STL"
mesh = pv.read(file_path)

# Print the coordinates of the points in the mesh (print selectively)
print(mesh.points)  # This shows all points; can inspect them one by one

# Visualize the original mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightblue', show_edges=True)
plotter.add_axes()  
plotter.show_grid()  



# Callback function to print coordinates of picked points
def print_point_coords(picked_point, plotter):
    print("Picked point coordinates:", picked_point)

# Enable point picking with callback, use_picker not mesh_picker
plotter.enable_point_picking(callback=print_point_coords, show_message=True, use_picker=True)

plotter.show()

# Define the fixed points based on approximate locations
fixed_points_indices = []  # Indices for points in Interfaces 2-5 (to fix)


#[154.16644287  22.53479195  52.67448807] en haut à gauche
#[153.5009613   48.94762039  30.06476021] en bas à gauche
#[23.97496986 25.72593307 49.6015358 ] en haut à droite
#[24.73397636 63.79221725 16.36836243] en bas à droite
#[78.15579224 26.76436424 85.2099762 ] 1 droite
#[100.38079071  31.54273987  83.92961884] 1 gauche



#Coordinates found by clicking on the plotter in the area of the interface
for i, point in enumerate(mesh.points):
    if np.allclose(point, [154.16644287,  22.53479195,  52.67448807], atol=5):  #Interface 5
        fixed_points_indices.append(i)
    elif np.allclose(point, [153.5009613,   48.94762039,  30.06476021], atol=5):  # Interface 4
        fixed_points_indices.append(i)
    elif np.allclose(point, [23.97496986, 25.72593307, 49.6015358 ], atol=5):  # Interface 3
        fixed_points_indices.append(i)
    elif np.allclose(point, [24.73397636, 63.79221725, 16.36836243], atol=5):  # Interface 2
        fixed_points_indices.append(i)

# Define points under load (Interface 1)
load_points_indices = []  # Indices for points on Interface 1

for i, point in enumerate(mesh.points):
    if np.allclose(point, [78.15579224, 26.76436424, 85.2099762 ], atol=5) or np.allclose(point, [100.38079071,  31.54273987,  83.92961884], atol=5):  # Interface 1
        load_points_indices.append(i)

# Apply load (no actual physics solver here, just visualization)
for idx in load_points_indices:
    mesh.points[idx] += [0, 0, -5]  # Apply a small load effect

# Visualize the simulated mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='salmon', show_edges=True)
plotter.add_axes()
plotter.show()


vtk_file_path = "bracket_mesh.vtk"
mesh.save(vtk_file_path)  # Save as VTK

#Read the VTK file for processing
converted_mesh = meshio.read(vtk_file_path)

xdmf_file_path = "bracket_mesh.xdmf"
meshio.write(xdmf_file_path, converted_mesh, file_format="xdmf")



#Load mesh into SfePy
sfepy_mesh = SfePyMesh.from_file(xdmf_file_path)
domain = FEDomain("domain", sfepy_mesh)

# Define Field and Problem
field = Field.from_args("displacement", np.float64, "vector", domain, approx_order=1)
u = FieldVariable("u", "unknown", field)
v = FieldVariable("v", "test", field, primary_var_name="u")

# Material properties for steel
E, nu = 210e9, 0.3  # Young's modulus and Poisson's ratio
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define problem formulation
integral = Integral("i", order=3)
t1 = Term.new("dw_lin_elastic_iso(m.lmbda, m.mu, v, u)", integral, domain, m={"lmbda": lmbda, "mu": mu}, v=v, u=u)

# Define load condition (Neumann BC)
load_value = np.array([0.0, 0.0, -1000.0])
load = Term.new("dw_surface_ltr", integral, domain, load=load_value, primary_var_name="u")

# Define Essential boundary conditions (Dirichlet BC)
# (Assuming placeholder fixed points)
fixed_points = [
    [154.16644287, 22.53479195, 52.67448807],
    [153.5009613, 48.94762039, 30.06476021],
    [23.97496986, 25.72593307, 49.6015358],
    [24.73397636, 63.79221725, 16.36836243]
]
ebcs = Conditions([
    EssentialBC("fixed", domain.get_region("vertices by bounding_box", (fixed_points,), select="vertex"), {"u.all": 0.0})
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

# Solve the problem
status = pb.solve()
print("Problem solved with status:", status)



# After solving the problem, visualize the displacement field using PyVista

# Load the saved VTK file or XDMF file
mesh = pv.read("bracket_mesh.vtk")  # Load the mesh data from the VTK file

# Assuming `displacement` is the result from your SfePy solution
# Set up the displacement field (you can adjust this based on your solution output)

# Retrieve the displacement field at each node
displacement = u.evaluate_at_nodes()  # Evaluate the displacement field
mesh.point_data["displacement"] = displacement  # Assign to mesh for visualization

# Initialize a PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)

# Add arrows to show the displacement vectors
plotter.add_arrows(mesh.points, displacement, mag=0.1, color="red")  # Adjust 'mag' for scale

# Show the plot with displacement
plotter.add_axes()  # Add coordinate axes
plotter.show_grid()  # Show grid for better orientation
plotter.show()