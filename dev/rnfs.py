#!/usr/bin/env python

import os
import torch
import torch.nn as nn
from data.dataloader import NSF_dataset, SFDataset4D, DATA_PATH
from loss.flow import DT, SmoothnessLoss
from ops.metric import scene_flow_metrics
from ops.transform import xyz_axis_angle_to_matrix, matrix_to_xyz_axis_angle
import pandas as pd
from loss.path import path_smoothness
from tqdm import tqdm
from mayavi import mlab
from vis.mayavi_interactive import draw_coord_frames


class PoseTransform(torch.nn.Module):
    """
    Pose transform layer
    Works as a differentiable transformation layer to fit rigid ego-motion
    """
    def __init__(self, device='cpu'):
        super().__init__()
        # If not working in sequences, use LieTorch
        self.xyza = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=True, device=device))

    def construct_pose(self):
        pose = xyz_axis_angle_to_matrix(self.xyza)
        return pose

    def forward(self, pc):
        pose = xyz_axis_angle_to_matrix(self.xyza)
        deformed_pc = pc @ pose[:, :3, :3].transpose(1, 2) + pose[:, :3, -1]
        return deformed_pc


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class RigidNeuralPrior(torch.nn.Module):
    """
    Neural Prior with Rigid Transformation, takes only point cloud t=1 on input and returns flow and rigid flow (ego-motion flow)
    """
    def __init__(self, pc1, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        self.RigidTransform = PoseTransform(device=pc1.device)
        # testing refinement
        self.Refinement = torch.nn.Parameter(torch.randn(pc1.shape, requires_grad=True))
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

    def forward(self, x):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """

        deformed_pc = self.RigidTransform(x)
        rigid_flow = deformed_pc - x

        # x = self.Refinement / 10 + rigid_flow
        x = self.nn_layers[0](deformed_pc)
        for layer in self.nn_layers[1:]:
            # deformed_pc = layer(deformed_pc)
            x = layer(x)

        final_flow = x + rigid_flow

        return final_flow, rigid_flow

    def initialize(self):
        self.apply(init_weights)


def train():
    # Main training loop
    dataset = NSF_dataset()
    # n_clouds = 10
    # dataset = SFDataset4D(root_dir=os.path.join(DATA_PATH, 'sceneflow'),
    #                       dataset_type='argoverse', data_split='train4', n_frames=n_clouds)
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    n_flow_iters = 10
    n_path_iters = 10

    # path for ego-motion between the two frames
    path = torch.zeros((1, 6), requires_grad=True, device=device)

    for frame_id, data in tqdm(enumerate(dataset)):
        pc1, pc2, gt_flow = data['pc1'], data['pc2'], data['gt_flow']

        pc1 = pc1.to(device)
        pc2 = pc2.to(device)

        # One model per both frames
        model = RigidNeuralPrior(pc1, 3, 128, 'relu', 8).to(device)

        optimizer = torch.optim.Adam([model.RigidTransform.xyza], lr=0.008)
        DT_layer = DT(pc1, pc2)

        # Flow and pose optimization
        for e in range(n_flow_iters):
            pred_flow, rigid_flow = model(pc1)

            dt_loss, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)
            # rigid_dt_loss, _ = DT_layer.torch_bilinear_distance(pc1 + rigid_flow)

            # Loss
            loss = dt_loss# + rigid_dt_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {e:03d},"
                  f"NN Loss: {dt_loss.item():.3f}"
                  #f"\t Rigid Loss: {rigid_dt_loss.item():.3f}"
            )

        # update path with new pose
        xyza = model.RigidTransform.xyza
        path = torch.cat([path, xyza], dim=0)
        print(path)
        if len(path) > 3:
            for _ in range(n_path_iters):
                # optimizer.zero_grad()
                path_loss = path_smoothness(path)
                print(f"Path Loss: {path_loss.item()}")
                path_loss.backward()
                optimizer.step()

            # # visualize path
            # with torch.no_grad():
            #     # visualize traj in mayavi
            #     mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
            #     path_poses = xyz_axis_angle_to_matrix(path)
            #     draw_coord_frames(path_poses, scale=0.1)
            #     mlab.show()

        # Computation of sceneflow metrics
        epe, accs, accr, angle, outlier = scene_flow_metrics(pred_flow, gt_flow.to(device))

        # Display metrics
        pd.set_option("display.precision", 3)
        metric_df = pd.DataFrame([epe, accs, accr, angle, outlier],
                                 index=['epe', 'accs', 'accr', 'angle', 'outlier']).T
        print(metric_df.describe())


def rigid_transform_test():
    import open3d as o3d
    import numpy as np

    dataset = NSF_dataset()
    data = next(iter(dataset))
    pc = data['pc1']

    with torch.no_grad():
        model = RigidNeuralPrior(pc, 3, 128, 'relu', 8)
        model.RigidTransform.xyza = torch.nn.Parameter(torch.tensor([[0, 0, 10, 0, 0, np.pi/4]], dtype=pc.dtype))
        deformed_pc = model.RigidTransform(pc)

        pc = pc.numpy().squeeze()
        deformed_pc = deformed_pc.numpy().squeeze()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color([1, 0, 0])

        deformed_pcd = o3d.geometry.PointCloud()
        deformed_pcd.points = o3d.utility.Vector3dVector(deformed_pc)
        deformed_pcd.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([pcd, deformed_pcd])


def main():
    train()
    # rigid_transform_test()


if __name__ == '__main__':
    main()
