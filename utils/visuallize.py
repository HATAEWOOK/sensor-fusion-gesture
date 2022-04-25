from email.mime import base
import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, path, epoch, ibx):
            self.path = path
            file = f'E{epoch}_{ibx}_'
            verts = os.path.join(path, file + 'vert' + '.txt')
            faces = os.path.join(path, file + 'faces' + '.txt')
            joints = os.path.join(path, file + 'joint' + '.txt')
            self.verts = np.loadtxt(verts, dtype=float, delimiter=' ', skiprows=0)
            self.faces = np.loadtxt(faces, dtype=float, delimiter=' ', skiprows=0)
            self.joints = np.loadtxt(joints, dtype=float, delimiter=' ', skiprows=0)

    def joint_mesh(self, data=None, radius = 2):
        i = 0
        mesh_joints = []
        if data is None:
            joint_list = self.joints
        else:
            joint_list = data
        for j in joint_list:
            p = o3d.geometry.TriangleMesh.create_sphere(radius)
            p.compute_vertex_normals()
            if i == 0:
                p.paint_uniform_color([1,0,0])
            elif i > 0 and i <= 4:
                p.paint_uniform_color([1,1,0])
            elif i > 4 and i <= 8:
                p.paint_uniform_color([0,1,0])
            elif i > 8 and i <= 12:
                p.paint_uniform_color([0,1,1])
            elif i > 12 and i <= 16:
                p.paint_uniform_color([0,0,1])
            elif i > 16 and i <= 20:
                p.paint_uniform_color([0,0,0])
            i += 1
            p.translate(j)
            mesh_joints.append(p)
        
        return mesh_joints

    def hand_mesh(self):
        mesh_hand = o3d.geometry.TriangleMesh()
        mesh_hand.vertices = o3d.utility.Vector3dVector(self.verts)
        mesh_hand.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh_hand.compute_vertex_normals()

        return [mesh_hand]

    def mesh_show(self,mesh_list):
        return o3d.visualization.draw_geometries(mesh_list)

if __name__ == '__main__':
    path = '/root/sensor-fusion-gesture/ckp/FreiHAND/results0'
    v = Visualizer(path, 5, 400)

    j = v.hand_mesh()
    v.mesh_show(j)
