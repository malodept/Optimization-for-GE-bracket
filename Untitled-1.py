import pyvista as pv
import numpy as np

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

# Visualize the "optimized" mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='salmon', show_edges=True)
plotter.add_axes()
plotter.show()


