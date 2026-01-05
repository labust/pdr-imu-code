import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sin, cos, atan2


data = np.loadtxt("right_leg_imu_data.csv", delimiter=",", skiprows=1)

ax, ay, az = data[:, 1], data[:, 2], data[:, 3]
gx, gy, gz = data[:, 4], data[:, 5], data[:, 6]
mx, my, mz = data[:, 7], data[:, 8], data[:, 9]

gx = np.deg2rad(gx)
gy = np.deg2rad(gy)
gz = np.deg2rad(gz)

timestamps = data[:, 0]
differences = []
for index, t in enumerate(timestamps):
    if len(timestamps) - 1 <= index:
        break
    
    difference = timestamps[index + 1] - t
    if difference < 60:
        differences.append(difference)
differences = np.array(differences)
dt = differences.mean() / 1000

q = 0.99 # 0-1, što je q blizi 1 to je veci fokus na gyro, sto je manje to je veci fokus na akcelerometar

A_roll = np.atan2(ay, az)
A_pitch = np.atan2(-ax, np.sqrt(ay ** 2 + az ** 2))

roll = []
pitch = []
yaw = []
roll_prev = A_roll[0]
pitch_prev = A_pitch[0]
yaw_prev = 0
for i in range(len(A_roll)):
    r = A_roll[i] * (1 - q) + (roll_prev + gx[i] * dt) * q
    p = A_pitch[i] * (1 - q) + (pitch_prev + gy[i] * dt) * q
    
    mx2 = mx[i]*cos(p) + mz[i]*sin(p)
    my2 = mx[i]*sin(r)*sin(p) + my[i]*cos(r) - mz[i]*sin(r)*cos(p)
    m_yaw = atan2(-my2, mx2)
    
    y = m_yaw * (1 - q) + (yaw_prev + gz[i] * dt) * q
    
    roll_prev = r
    pitch_prev = p
    yaw_prev = y
    
    roll.append(r)
    pitch.append(p)
    yaw.append(y)

roll = np.degrees(np.array(roll))
pitch = np.degrees(np.array(pitch))
yaw = np.degrees(np.array(yaw))

fusion = np.transpose(np.vstack([roll, pitch, yaw]))

header = 'roll, pitch, yaw'
np.savetxt('fusion_right_leg_bedra_hodanje_krug.csv', fusion, delimiter=',', header=header, comments='')

# === GPT GENERATED ===
frames = len(roll)
# roll  = 30 * np.sin(0.1*np.arange(frames)) 
# pitch = 20 * np.sin(0.05*np.arange(frames))
# yaw   = 45 * np.sin(0.03*np.arange(frames))

# Cube definition (centered at origin, size=1)
cube_vertices = np.array([[-0.8, -0.1, -0.3],
                          [ 0.8, -0.1, -0.3],
                          [ 0.8,  0.1, -0.3],
                          [-0.8,  0.1, -0.3],
                          [-0.8, -0.1,  0.3],
                          [ 0.8, -0.1,  0.3],
                          [ 0.8,  0.1,  0.3],
                          [-0.8,  0.1,  0.3]])

# Cube faces (for Poly3DCollection)
faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],
         [2,3,7,6],[0,3,7,4],[1,2,6,5]]

# Rotation matrix from roll, pitch, yaw (degrees)
def rotation_matrix(r, p, y):
    r, p, y = np.radians([r, p, y])
    Rx = np.array([[1,0,0],
                   [0,np.cos(r),-np.sin(r)],
                   [0,np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],
                   [0,1,0],
                   [-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],
                   [np.sin(y), np.cos(y),0],
                   [0,0,1]])
    return Rz @ Ry @ Rx

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1])

cube_poly = Poly3DCollection([], facecolors='cyan', edgecolors='k', linewidths=1, alpha=0.7)
ax.add_collection3d(cube_poly)

def update(frame):
    if(frame == 0):
        print("Started")
    R = rotation_matrix(roll[frame], pitch[frame], yaw[frame])
    rotated_vertices = cube_vertices @ R.T
    verts = [rotated_vertices[face] for face in faces]
    cube_poly.set_verts(verts)
    return cube_poly,

animation = FuncAnimation(fig, update, frames=frames, interval=dt * 1000, blit=False)
plt.show()