# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
# Vassilis Choutas <https://ps.is.tuebingen.mpg.de/employees/vchoutas>
# 2018.01.02

import os

# if 'GPU_DEVICE_ORDINAL' in os.environ:
#     print('You are rendering on the cluster')
#     

import numpy as np
from body_visualizer.tools.vis_tools import colors
import trimesh
# os.environ["PYOPENGL_PLATFORM"] = "glfw"
try:
    del os.environ['PYOPENGL_PLATFORM']
except:
    pass
import pyrender
import sys
import cv2
from pyrender import Viewer

__all__ = ['MeshViewer']

class MeshViewer(object):

    def __init__(self, width=1200, height=800, use_offscreen=True):
        #super().__init__()

        self.width, self.height = width, height
        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3.0])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(4.)
        else:
            self.viewer = Viewer(self.scene, use_raymond_lighting=True, viewport_size=self.figsize, cull_faces=False, run_in_thread=True)

    def set_background_color(self, color=colors['white']):
        self.scene.bg_color = color

    def set_cam_trans(self, trans= [0, 0, 3.0]):
        if isinstance(trans, list): trans = np.array(trans)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = trans
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_meshes(self, meshes, group_name='static', poses=[]):
        for node in self.scene.get_nodes():
            if node.name is not None and '%s-mesh'%group_name in node.name:
                self.scene.remove_node(node)

        if len(poses) < 1:
            for mid, mesh in enumerate(meshes):
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid))
        else:
            for mid, iter_value in enumerate(zip(meshes, poses)):
                mesh, pose = iter_value
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid), pose)

    def set_static_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='static', poses=poses)
    def set_dynamic_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='dynamic', poses=poses)

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity = 1.0):
        if not self.use_offscreen:
            sys.stderr.write('Interactive viewer already uses raymond lighting!\n')
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)

    def render(self, render_wireframe=None, RGBA=False):
        from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if RGBA: flags |=  RenderFlags.RGBA
        if render_wireframe is not None and render_wireframe==True:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        return color_img

    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write('Currently saving snapshots only works with off-screen renderer!\n')
            return
        color_img = self.render()
        cv2.imwrite(fname, color_img)

if __name__ == '__main__':
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from human_body_prior.body_model.body_model import BodyModel
    from supercap.marker_layout_detection.tools import marker_layout_as_points, equal_aspect_ratio, visualize3DData

    bodymodel_fname = '/ps/project/common/moshpp/smplx/unlocked_head/neutral/model.npz'
    body = BodyModel(bodymodel_fname)()

    superset_fname = '/ps/project/supercap/support_files/marker_layouts/superset_smplx_95.json'
    superset_data = marker_layout_as_points(c2c(body.v[0]), c2c(body.f))(superset_fname)

    markers = superset_data['markers']
    mv = MeshViewer(use_offscreen=False)
    body_v = c2c(body.v[0])
    faces = c2c(body.f)
    n_verts = body_v.shape[0]
    body_mesh = trimesh.Trimesh(vertices=body_v, faces=faces, vertex_colors=np.tile(colors['grey'], (n_verts, 1)))
    mv.set_dynamic_meshes([body_mesh])
    #clicked_markers = visualize3DData(markers, superset_data['labels'], body_verts = c2c(body.v[0]))