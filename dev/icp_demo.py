#! /usr/bin/env python

import os
from data.PATHS import DATA_PATH
from data.dataloader import SFDataset4D
from mayavi import mlab
from vis.mayavi_interactive import draw_coord_frames
from matplotlib import pyplot as plt
from ops.filters import filter_range, filter_grid
import torch
from loss.icp import point_to_point_dist
from loss.path import path_smoothness
from ops.transform import xyz_axis_angle_to_matrix, matrix_to_xyz_axis_angle


def transform_cloud(cloud, pose):
    assert isinstance(cloud, torch.Tensor)
    assert isinstance(pose, torch.Tensor)
    assert cloud.ndim == 2 and cloud.shape[1] == 3
    assert pose.shape == (4, 4)
    return torch.matmul(cloud, pose[:3, :3].T) + pose[:3, 3][None]


def demo(n_clouds=10, pose_noise_level=0., n_iters=1, vis_scene=True):
    ds = SFDataset4D(dataset_type='waymo', n_frames=n_clouds)
    i = 0
    poses12 = ds[i]['relative_pose']
    clouds = ds[i]['pc1']
    masks = ds[i]['padded_mask_N']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # construct path from relative poses
    pose = torch.eye(4)[None]
    poses = pose.clone()
    for i in range(len(poses12)):
        pose = pose @ torch.linalg.inv(poses12[i])
        poses = torch.cat([poses, pose], dim=0)

    poses_gt = poses.clone()
    # add noise
    poses[:, :3, 3] += torch.randn_like(poses[:, :3, 3]) * pose_noise_level

    clouds_list = []
    for i in range(len(clouds)):
        cloud = clouds[i]
        cloud = cloud[masks[i]]
        # filter point cloud
        # cloud = filter_depth(cloud, min=1., max=25.0)
        # cloud = filter_grid(cloud, grid_res=0.3)
        clouds_list.append(cloud)

    # put on device
    clouds = [cloud.to(device) for cloud in clouds_list]
    poses_gt = poses_gt.to(device)
    poses = poses.to(device)

    xyzas = matrix_to_xyz_axis_angle(poses)
    xyzas_gt = matrix_to_xyz_axis_angle(poses_gt)
    xyzas.requires_grad = True

    # define optimizer
    optimizer = torch.optim.Adam([xyzas], lr=0.01)

    plt.figure(figsize=(20, 5))
    if vis_scene:
        mlab_fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    path_loss = None
    for it in range(n_iters):
        # transform point clouds to the same frame (of a first point cloud)
        clouds_transformed = []
        for i in range(len(clouds)):
            xyza = xyzas[i]
            pose = xyz_axis_angle_to_matrix(xyza[None]).squeeze()
            # transform point cloud
            clouds_transformed.append(transform_cloud(clouds[i], pose))

        # icp loss
        icp_loss = point_to_point_dist(clouds_transformed, icp_inlier_ratio=0.9)
        print('ICP loss: %.3f [m]' % icp_loss.item())

        # total loss
        loss = icp_loss.clone()

        if len(xyzas) > 3:
            # path smoothness loss
            path_loss = path_smoothness(xyzas)
            print('Path smoothness loss: %.3f' % path_loss.item())
            loss += path_loss

        # metrics: trajectory difference
        # xyz_diff = torch.norm(xyzas[:, :3] - xyzas_gt[:, :3], dim=1).mean(dim=0)
        x_diff = torch.square(xyzas[:, 0] - xyzas_gt[:, 0]).mean()
        y_diff = torch.square(xyzas[:, 1] - xyzas_gt[:, 1]).mean()
        z_diff = torch.square(xyzas[:, 2] - xyzas_gt[:, 2]).mean()
        rot_diff = torch.norm(xyzas[:, 3:] - xyzas_gt[:, 3:], dim=1).mean()

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualize
        # plot ICP loss
        plt.subplot(1, 4, 1)
        plt.ylabel('ICP point to point loss')
        plt.xlabel('Iterations')
        plt.plot(it, icp_loss.item(), '.', color='b')
        plt.grid(visible=True)

        # plot path smoothness loss
        if len(xyzas) > 3 and path_loss is not None:
            plt.subplot(1, 4, 2)
            plt.ylabel('Path smoothness loss')
            plt.xlabel('Iterations')
            plt.plot(it, path_loss.item(), '.', color='b')
            plt.grid(visible=True)

        # plot translation difference
        plt.subplot(1, 4, 3)
        plt.ylabel('Translation difference')
        plt.xlabel('Iterations')
        # plt.plot(it, xyz_diff.item(), '.', color='b')
        plt.plot(it, x_diff.item(), '.', color='r')
        plt.plot(it, y_diff.item(), '.', color='g')
        plt.plot(it, z_diff.item(), '.', color='b')
        plt.grid(visible=True)

        # plot rotation difference
        plt.subplot(1, 4, 4)
        plt.ylabel('Rotation difference')
        plt.xlabel('Iterations')
        plt.plot(it, rot_diff.item(), '.', color='b')
        plt.grid(visible=True)

        plt.draw()
        plt.pause(0.01)

        if vis_scene and (it == 0 or it == n_iters - 1):
            with torch.no_grad():
                mlab.clf()
                # visualize path
                poses = xyz_axis_angle_to_matrix(xyzas)
                draw_coord_frames(poses.cpu()[::5], scale=0.5)
                mlab.plot3d(poses[:, 0, 3].cpu(), poses[:, 1, 3].cpu(), poses[:, 2, 3].cpu(),
                            color=(0, 1, 0), tube_radius=0.04)
                # visualize gt path
                draw_coord_frames(poses_gt.cpu()[::5], scale=0.2)
                mlab.plot3d(poses_gt[:, 0, 3].cpu(), poses_gt[:, 1, 3].cpu(), poses_gt[:, 2, 3].cpu(),
                            color=(0, 0, 1), tube_radius=0.04)

                # # visualize global cloud
                global_cloud = torch.cat(clouds_transformed, dim=0).cpu().numpy()
                global_cloud = filter_grid(global_cloud, grid_res=0.2)
                mlab.points3d(global_cloud[:, 0], global_cloud[:, 1], global_cloud[:, 2],
                              color=(0, 1, 0), scale_factor=0.1)

                # set up view point
                # mlab.view(azimuth=0, elevation=30, distance=50, focalpoint=(0, 0, 0))
                mlab.show()
                # mlab_fig.scene._lift()


def main():
    demo(n_clouds=50, pose_noise_level=0.2, n_iters=200, vis_scene=True)


if __name__ == '__main__':
    main()
