""" 3D plot script for spherical coordinates
"""

__author__ = 'Tinatin Baratashvili'
__email__ = 'tinatin.baratashvili@kuleuven.be'


import numpy as np
import matplotlib.pyplot as plt


def spherical_to_cartesian(r, phi, theta):
    """ This function converts spherical coordinates to spherical coordinates.
    Input:
        r (floar): r coordinate (length) radius
        phi (float): phi coordinate (radians) zenith
        theta (float): theta coordinate (radians) azimuth
    Returns:
        Cartesian coordinates (float): x, y, z
    """
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return x, y, z


assert spherical_to_cartesian(1,0,0) == (0,0,1.0) and np.allclose(spherical_to_cartesian(1, np.pi, np.pi),(0, 0, -1.0), 1.e-2), "it's wrong"
assert np.allclose(spherical_to_cartesian(1, 2*np.pi, 2*np.pi),(0, 0, 1.0), 1.e-2) and np.allclose(spherical_to_cartesian(1, -np.pi, -2*np.pi),(0, 0, 1.0), 1.e-2), "it's wrong"
assert np.allclose(spherical_to_cartesian(1, 2*np.pi, 2*np.pi),(0, 0, 1.0), 1.e-2), "it's wrong"

def plot_3d_data():
    """
    This code covnerts spherical coordinates to cartesian
    coordiantes and makes 3d plot in cartesian coordinates
    """
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    r = np.linspace(0, 1)
    theta = np.linspace(0, 2*np.pi)
    phi = np.linspace(0, 2*np.pi)
    x,y,z = spherical_to_cartesian(r, phi, theta)
    axes.plot(x,y,z)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    plt.show()

plot_3d_data()

# print (vector_coordinate)
# print (x_coord, y_coord, z_coord)
