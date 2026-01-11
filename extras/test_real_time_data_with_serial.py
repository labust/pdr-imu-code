"""
    Pedestrian Dead Reckoning
    11 Jan 2026
    Python script to test sensor fusion algorithms and real time communication 
"""

import serial
from math import atan2, radians, pi, sqrt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

t0 = theta = phi = 0.0

cube_vertices = np.array([[-0.3, -0.8, -0.1],
                          [ 0.3, -0.8, -0.1],
                          [ 0.3,  0.8, -0.1],
                          [-0.3,  0.8, -0.1],
                          [-0.3, -0.8,  0.1],
                          [ 0.3, -0.8,  0.1],
                          [ 0.3,  0.8,  0.1],
                          [-0.3,  0.8,  0.1]])

faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],
         [2,3,7,6],[0,3,7,4],[1,2,6,5]]

def wrap(a):
    return (a + pi) % (2*pi) - pi

def rotation_matrix(r, p):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)

    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])

    return Ry @ Rx

def complementary_filter(raw_data: str):
    global t0, theta, phi
    split_data = raw_data.split(", ")
    if len(split_data) != 10:
        return 0.0, 0.0, 0.0
    
    t, ax, ay, az, gx, gy, gz, mx, my, mz = map(float, split_data)
    gx = radians(gx); gy = radians(gy); gz = radians(gz)
    
    dt = (t - t0) / 1000
    q = 0.7
    
    theta_measured_a = atan2(ax, sqrt(ay**2 + az**2))
    phi_measured_a = atan2(-ay, az)
    
    theta = wrap((theta + gy * dt) * q + theta_measured_a * (1 - q))
    phi = wrap((phi - gx * dt) * q + phi_measured_a * (1 - q))
    
    t0 = t
    
    return (theta, phi)

def read_serial(port, baudrate):
    plt.ion()
    
    fig = plt.figure()
    a_x = fig.add_subplot(111, projection='3d')
    a_x.set_xlim([-1,1])
    a_x.set_ylim([-1,1])
    a_x.set_zlim([-1,1])
    a_x.set_box_aspect([1,1,1])

    cube_poly = Poly3DCollection([], facecolors= ['yellow', 'cyan', 'red', 'cyan', 'cyan', 'blue'], edgecolors='k', linewidths=1, alpha=0.9)
    a_x.add_collection3d(cube_poly)
    
    rotated_vertices = cube_vertices
    verts = [rotated_vertices[face] for face in faces]
    cube_poly.set_verts(verts)
    
    with serial.Serial(port, baudrate, timeout=1) as ser:
        try:
            while True:
                data = ser.readline().decode().strip()
                if data:
                    theta, phi = complementary_filter(data)
                    R = rotation_matrix(theta, phi)
                    
                    rotated_vertices = cube_vertices @ R.T
                    verts = [rotated_vertices[face] for face in faces]
                    cube_poly.set_verts(verts)

                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                    
        except serial.SerialException as e:
            print(e)
        except KeyboardInterrupt:
            print("Program stopped!")

if __name__ == '__main__':
    read_serial('/dev/cu.usbmodem1101', 9600)
    