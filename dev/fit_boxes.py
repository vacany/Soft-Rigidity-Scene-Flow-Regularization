import os
from data.dataloader import SFDataset4D
import open3d as o3d
from loss.path import path_smoothness_batched, path_smoothness
from vis.open3d import visualize_points3D, visualize_poses, visualize_bbox, map_colors
from loss.flow import DT, GeneralLoss
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from loss.box import bboxes_coverage, bboxes_optimization, bboxes_iou
from ops.transform import matrix_to_xyz_yaw, xyz_yaw_to_matrix
from ops.filters import filter_grid, filter_box
from matplotlib import pyplot as plt
from pytorch3d.ops.knn import knn_points
from data.path_utils import rel_poses2traj


def get_motion_pc(data_i=0, device=torch.device('cuda:0')):
    fname = f'motion_pc_{data_i}.npy'
    if os.path.exists(fname):
        print(f'Loading motion point cloud {data_i}')
        # load motion point cloud
        motion_pc = np.load(fname)
        return motion_pc
    dataset = SFDataset4D(dataset_type='waymo', n_frames=3, only_first=False)
    data = dataset[data_i]
    pc = data['pc1']
    print(f'Generating motion point cloud {data_i}...')
    motion_pc = generate_motion_pc(pc, device=device)
    # save motion point cloud
    print('Saving motion point cloud as {}'.format(fname))
    np.save(fname, motion_pc)

    return motion_pc


def generate_motion_pc(pc, device=torch.device('cuda:0')):
    pc = pc.to(device)
    pc = [pc[i][pc[i, :, 2] > 0.3] for i in range(0, 3)]

    c_pc = pc[1].unsqueeze(0)
    b_pc = pc[2].unsqueeze(0)
    f_pc = pc[0].unsqueeze(0)

    # params
    forth_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
    back_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([forth_flow, back_flow], lr=0.008)
    # losses
    SM_loss = GeneralLoss(pc1=c_pc, K=16, max_radius=1)

    b_DT = DT(c_pc, b_pc)
    f_DT = DT(c_pc, f_pc)

    for i in range(500):
        # forth_dist, forth_nn, _ = knn_points(c_pc + forth_flow, f_pc, K=1, return_nn=True)
        # back_dist, back_nn, _ = knn_points(c_pc + back_flow, b_pc, K=1, return_nn=True)
        forth_dist, _ = f_DT.torch_bilinear_distance(c_pc + forth_flow)
        back_dist, _ = b_DT.torch_bilinear_distance(c_pc + back_flow)

        dist_loss = (forth_dist + back_dist).mean()
        smooth_loss = SM_loss(c_pc, forth_flow, f_pc) + SM_loss(c_pc, back_flow, f_pc)

        time_smooth = (forth_flow - (-back_flow)).norm(dim=2, p=1).mean()  # maybe magnitude of flow?

        loss = dist_loss + smooth_loss + time_smooth

        print(f'Iteration {i}, loss: {loss.item():.4f}, time_smooth: {time_smooth.item():.4f}, ')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    mos = (forth_flow[0].norm(dim=1, p=1) > 0.05).detach().cpu().numpy()  # mask of static and dynamic
    numpy_c_pc = c_pc.detach().cpu().numpy()
    numpy_forth_flow = forth_flow.detach().cpu().numpy()

    motion_pc = np.concatenate((numpy_c_pc[0, mos], numpy_forth_flow[0, mos]), axis=1)

    return motion_pc


def get_boxes(points, flow, ids):
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert flow.ndim == 2
    assert flow.shape[1] == 3

    # find points with the same id
    box_poses = []
    box_sizes = []
    for id in np.unique(ids):
        if id == -1:
            continue

        inst_points = points[ids == id]
        # xyz = np.mean(inst_points, axis=0)
        xyz = inst_points.min(axis=0) + (inst_points.max(axis=0) - inst_points.min(axis=0)) / 2
        inst_flows = flow[ids == id]
        flow_vector = np.mean(inst_flows, axis=0)
        # normalize flow vector
        flow_vector /= np.linalg.norm(flow_vector)
        # find yaw angle
        yaw = np.arctan2(flow_vector[1], flow_vector[0])
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0., 0., 1.]])
        # create box parameters
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = R
        # estimate box size
        inst_points_centered = (inst_points - xyz) @ R
        lhw = inst_points_centered.max(axis=0) - inst_points_centered.min(axis=0)
        # add to lists
        box_poses.append(pose)
        box_sizes.append(lhw)
    box_poses = np.asarray(box_poses)
    box_sizes = np.asarray(box_sizes)

    return box_poses, box_sizes


def visualize(box_poses, box_sizes, points, ids=None, vis=None):
    if isinstance(box_poses, torch.Tensor):
        box_poses = box_poses.detach().cpu().numpy()
    if isinstance(box_sizes, torch.Tensor):
        box_sizes = box_sizes.detach().cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    assert isinstance(box_poses, np.ndarray)
    assert isinstance(box_sizes, np.ndarray)
    assert isinstance(points, np.ndarray)
    assert isinstance(ids, np.ndarray) or isinstance(ids, list)
    assert box_poses.ndim == 3
    assert box_poses.shape[1:] == (4, 4)
    assert box_sizes.ndim == 2
    assert box_sizes.shape[1] == 3
    assert points.ndim == 2
    assert points.shape[1] == 3

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    for i in range(len(box_poses)):
        # add box geometry
        line_set = visualize_bbox(box_sizes[i], box_poses[i], vis=False)
        vis.add_geometry(line_set)

    # add point cloud geometry
    pcd = visualize_points3D(points, ids, vis=False)
    vis.add_geometry(pcd)
    vis.run()


def fit_boxes_to_cloud(n_iters=100, lr=0.1, grid_res=None, plot_metrics=False, show=True, data_i=80):
    print('Generating motion point cloud...')
    motion_pc = get_motion_pc(data_i=data_i)
    # cluster points
    ids = DBSCAN(eps=0.4, min_samples=10).fit_predict(motion_pc)  # id mask for dynamic only

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get boxes
    box_poses, box_sizes = get_boxes(motion_pc[..., :3], motion_pc[..., 3:], ids)

    points = motion_pc[..., :3]
    if grid_res is not None:
        points = filter_grid(points, grid_res=grid_res)

    # compute coverage
    n_inst = len(box_poses)
    n_time_stamps = 1
    box_poses = torch.as_tensor(box_poses).reshape(n_inst, n_time_stamps, 4, 4)
    box_sizes = torch.as_tensor(box_sizes).reshape(n_inst, n_time_stamps, 3)
    points = torch.as_tensor(points)
    # put to device
    box_poses = box_poses.to(device)
    box_sizes = box_sizes.to(device)
    points = points.to(device)

    # optimize bounding boxes to maximize coverage
    xyz_yaw = matrix_to_xyz_yaw(box_poses)
    xyz_yaw.requires_grad = True
    box_sizes.requires_grad = True
    optimizer = torch.optim.Adam([xyz_yaw, box_sizes], lr=lr)
    # optimizer = torch.optim.Adam([xyz_yaw], lr=lr)

    # Create a visualization object and window
    if show:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    if plot_metrics:
        plt.figure(figsize=(20, 5))

    for i in range(n_iters):
        box_poses = xyz_yaw_to_matrix(xyz_yaw)
        rewards, coverage_mask = bboxes_coverage(points, box_poses, torch.relu(box_sizes),
                                                 sigmoid_slope=5., return_mask=True, reduce_rewards=False)
        coverage = rewards.mean()

        # compute loss
        boxes_mean_volume = torch.relu(box_sizes).prod(dim=2).mean()  # boxes mean volume
        loss = -coverage + 0.1 * boxes_mean_volume
        print('iter {}, loss: {:.4f}, coverage: {:.4f}'.format(i, loss.item(), coverage.item()))

        # compute iou between neighboring boxes
        # if plot_metrics:
        #     with torch.no_grad():
        #         iou = bboxes_iou(box_poses, box_sizes)
        #         print('iou: ', iou.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualize
        if show:
            vis.clear_geometries()
            visualize(box_poses.squeeze(), torch.relu(box_sizes).squeeze(), points, rewards.detach().cpu().numpy(), vis=vis)
            # visualize(box_poses.squeeze(), torch.relu(box_sizes).squeeze(), points, ids, vis=vis)

        if plot_metrics:
            # plot loss
            plt.subplot(1, 4, 1)
            plt.plot(i, loss.item(), 'b.')
            plt.title('loss')
            plt.grid(visible=True)

            plt.subplot(1, 4, 2)
            plt.plot(i, coverage.item(), 'r.')
            plt.title('coverage')
            plt.grid(visible=True)

            plt.subplot(1, 4, 3)
            plt.plot(i, coverage_mask.float().mean().item(), 'g.')
            plt.title('Covered points ratio')
            plt.grid(visible=True)

            plt.subplot(1, 4, 4)
            plt.plot(i, boxes_mean_volume.item(), 'g.')
            # plt.plot(i, iou.item(), 'b.')
            plt.title('Boxes mean volume (g) and IoU (b)')
            plt.grid(visible=True)

            plt.pause(0.01)
            plt.draw()

    if show:
        # visualize final result
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        visualize(box_poses.squeeze(), box_sizes.squeeze(), points, rewards.detach().cpu().numpy(), vis=vis)
        # visualize(box_poses.squeeze(), torch.relu(box_sizes).squeeze(), points, ids, vis=vis)
        vis.destroy_window()
    if plot_metrics:
        plt.show()


def estimate_box_trajectories(data_i=0, n_frames=5):
    motion_pcs = []
    clusters = []
    box_poses = []
    box_sizes = []
    dataset = SFDataset4D(dataset_type='waymo', n_frames=n_frames, only_first=False)

    # construct path from relative poses
    poses12 = dataset[data_i]['relative_pose']
    ego_poses = rel_poses2traj(poses12).cpu().numpy()

    for k in range(data_i, data_i + n_frames, 1):
        motion_pc = get_motion_pc(data_i=k)

        # transform motion point cloud to the first frame
        p = ego_poses[k - data_i]
        cloud = motion_pc[:, :3] @ p[:3, :3].T + p[:3, 3]
        flow = motion_pc[:, 3:] @ p[:3, :3].T
        motion_pc = np.concatenate((cloud, flow), axis=1)
        motion_pcs.append(motion_pc)

        # cluster points
        ids = DBSCAN(eps=0.4, min_samples=10).fit_predict(motion_pc)  # id mask for dynamic only
        clusters.append(ids)

        # get boxes
        poses, sizes = get_boxes(motion_pc[..., :3], motion_pc[..., 3:], ids)
        box_poses.append(poses)
        box_sizes.append(sizes)

    points = torch.as_tensor(np.concatenate(motion_pcs, axis=0)[..., :3])
    print('Total number of points: ', len(points))
    flows = torch.as_tensor(np.concatenate(motion_pcs, axis=0)[..., 3:])
    clusters = np.concatenate(clusters, axis=0)
    assert len(points) == len(clusters)

    # plot point cloud and flow as normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(flows)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    box_poses_c = torch.as_tensor(box_poses[0])
    box_sizes_c = torch.as_tensor(box_sizes[0])
    # visualize(box_poses_c, box_sizes_c, points_c, ids_c)
    n_instances = len(box_poses_c)

    box_poses_f = torch.as_tensor(np.concatenate(box_poses[1:], axis=0))
    box_sizes_f = torch.as_tensor(np.concatenate(box_sizes[1:], axis=0))
    # visualize(box_poses_f, box_sizes_f, points_f, ids_f)

    # create feature tensors
    xyz_yaw_lhw_c = torch.cat((matrix_to_xyz_yaw(box_poses_c), box_sizes_c), dim=1)
    xyz_yaw_lhw_f = torch.cat((matrix_to_xyz_yaw(box_poses_f), box_sizes_f), dim=1)

    # find nearest neighbors from current frame to future frame
    dists, idx, nn = knn_points(xyz_yaw_lhw_c[None].float(), xyz_yaw_lhw_f[None].float(), K=n_frames-1, return_nn=True)
    idx = idx.squeeze()
    nn = nn.squeeze()
    xyz_yaw_lhw_nn = xyz_yaw_lhw_f[idx]
    assert torch.allclose(xyz_yaw_lhw_nn.float(), nn)

    # get trajectories of boxes from current frame to future
    trajes = torch.cat((xyz_yaw_lhw_c[:, None], xyz_yaw_lhw_nn), dim=1)
    assert trajes.shape == (n_instances, n_frames, 7)

    box_poses = xyz_yaw_to_matrix(trajes[:, :, :4])
    box_sizes = trajes[:, :, 4:]
    assert box_poses.shape == (n_instances, n_frames, 4, 4)
    assert box_sizes.shape == (n_instances, n_frames, 3)

    # filter out boxes trajectories with low smoothness
    smoothness_cost = path_smoothness_batched(box_poses, reduce=False)
    mask = smoothness_cost < torch.quantile(smoothness_cost, 0.6)
    box_poses = box_poses[mask]
    box_sizes = box_sizes[mask]
    print('After filtering by smoothness, {} boxes remained'.format(len(box_poses)))

    # filter out boxes based on high width to height ratio
    wh_ratio = box_sizes[..., 1] / box_sizes[..., 2]
    wh_ratio = wh_ratio.mean(dim=1)
    mask = wh_ratio < torch.quantile(wh_ratio, 0.8)
    box_poses = box_poses[mask]
    box_sizes = box_sizes[mask]
    n_instances = len(box_poses)
    print('After filtering by width to height ratio, {} boxes remained'.format(n_instances))

    return box_poses, box_sizes, points, clusters


def optimize_boxes(box_poses, box_sizes, points, clusters=None,
                   n_iters=100, lr=0.1, grid_res=None,
                   volume_weight=1.,
                   device=torch.device('cuda:0')):
    n_instances = len(box_poses)
    n_time_stamps = len(box_poses[0])
    assert box_poses.shape == (n_instances, n_time_stamps, 4, 4)
    assert box_sizes.shape == (n_instances, n_time_stamps, 3)
    assert points.shape[1] == 3

    # boxes optimization
    xyz_yaws = matrix_to_xyz_yaw(box_poses)
    # put tensors to device
    xyz_yaws = xyz_yaws.to(device)
    box_sizes = box_sizes.to(device)
    if grid_res:
        grid_mask = filter_grid(points, grid_res, only_mask=True)
        points = points[grid_mask]
        clusters = clusters[grid_mask]
    points = points.to(device)
    xyz_yaws.requires_grad = True
    box_sizes.requires_grad = True
    optimizer = torch.optim.Adam([xyz_yaws, box_sizes], lr=lr)
    # optimizer = torch.optim.Adam([xyz_yaws], lr=lr)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(n_iters):
        box_poses = xyz_yaw_to_matrix(xyz_yaws)
        rewards = bboxes_coverage(points, box_poses, box_sizes, sigmoid_slope=4., reduce_rewards=False)
        coverage = rewards.mean()
        coverage = torch.clamp(coverage, min=1e-6)

        # compute boxes volume
        boxes_mean_volume = box_sizes.abs().prod(dim=2).mean()

        # trajectory smoothness
        smoothness_cost = path_smoothness_batched(box_poses)

        # compute loss
        loss = 1. / coverage + volume_weight * boxes_mean_volume + smoothness_cost
        print('iter {}, '
              'loss: {:.4f}, '
              'coverage: {:.4f}, '
              'smoothness cost: {:.4f}, '
              'boxes mean volume: {:.2f}'.format(i,
                                                 loss.item(),
                                                 coverage.item(),
                                                 smoothness_cost.item(),
                                                 boxes_mean_volume.item()))
        with torch.no_grad():
            vis.clear_geometries()
            # path for each instance
            for inst_i in range(n_instances):
                p = box_poses.cpu().numpy()[inst_i]
                color = np.random.random(3)
                colors = np.asarray([color for _ in range(len(p) - 1)])
                line_set = visualize_poses(p, colors, vis=False)
                vis.add_geometry(line_set)
            # visualize(box_poses.view((-1, 4, 4)), box_sizes.view((-1, 3)), points, rewards, vis=vis)
            visualize(box_poses.view((-1, 4, 4)), box_sizes.view((-1, 3)), points, clusters, vis=vis)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # visualize final result
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.clear_geometries()
        # path for each instance
        for inst_i in range(n_instances):
            p = box_poses.cpu().numpy()[inst_i]
            color = np.random.random(3)
            colors = np.asarray([color for _ in range(len(p) - 1)])
            line_set = visualize_poses(p, colors, vis=False)
            vis.add_geometry(line_set)
        # visualize(box_poses.view((-1, 4, 4)), box_sizes.view((-1, 3)), points, rewards, vis=vis)
        visualize(box_poses.view((-1, 4, 4)), box_sizes.view((-1, 3)), points, clusters, vis=vis)
        vis.destroy_window()


def estimate_and_optimize_box_trajectories(data_i=0, n_frames=5, n_iters=100, lr=0.1, grid_res=None, volume_weight=0.01):
    # estimate box trajectories
    box_poses, box_sizes, points, clusters = estimate_box_trajectories(data_i=data_i, n_frames=n_frames)
    # optimize boxes
    optimize_boxes(box_poses, box_sizes, points, clusters,
                   n_iters=n_iters, lr=lr, volume_weight=volume_weight, grid_res=grid_res)


def fit_box(lr=0.1, n_iters=100, sigmoid_slope=5., vis=True):
    import scipy

    np.random.seed(0)
    points = np.load('car_points.npy')
    R = scipy.spatial.transform.Rotation.from_euler('z', np.pi / 4).as_matrix()
    points = points @ R.T
    # cloud point weights
    # point_weights = np.linalg.norm(points, axis=1)
    point_weights = np.ones(len(points)) * 0.8
    mask = np.logical_or(points[:, 2] < -0.2, points[:, 2] > 0.2)
    point_weights[mask] = 0.2
    point_weights /= point_weights.max()
    print('Ration of points with zero weight: ', (point_weights == 0).sum() / len(point_weights))

    points_centered = (points - points.mean(axis=0)) @ R
    lwh = points_centered.max(axis=0) - points_centered.min(axis=0)
    lwh0 = lwh.copy()
    volume0 = lwh.prod()

    pose = np.eye(4)
    center = points.min(axis=0) + (points.max(axis=0) - points.min(axis=0)) / 2
    # center = points.mean(axis=0)
    pose[:3, 3] = center
    pose[:3, :3] = R
    print('pose:\n', pose)

    if vis:
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()

    plt.figure()
    plt.title('Coverage dependence on volume')
    plt.xlabel('Volume scale: V / V0')
    plt.ylabel('Coverage')
    plt.grid(visible=True)

    # optimize box
    box_pose = torch.as_tensor(pose)
    box_size = torch.as_tensor(lwh)
    points = torch.as_tensor(points)
    point_weights = torch.as_tensor(point_weights)

    xyz_yaw = matrix_to_xyz_yaw(box_pose[None]).squeeze()
    xyz_yaw.requires_grad = True
    box_size.requires_grad = True

    optimizer = torch.optim.Adam([xyz_yaw, box_size], lr=lr)
    for it in range(n_iters):
        box_pose = xyz_yaw_to_matrix(xyz_yaw[None]).squeeze()

        # compute coverage
        positive_coverage = bboxes_coverage(points, box_pose[None][None], box_size[None][None], weights=point_weights, sigmoid_slope=sigmoid_slope)
        negative_coverage = bboxes_coverage(points, box_pose[None][None], box_size[None][None], weights=1.-point_weights, sigmoid_slope=sigmoid_slope)
        coverage = positive_coverage - negative_coverage
        # compute volume
        volume = torch.relu(box_size).prod()

        # loss = 1. / torch.clamp(coverage, min=1e-6) + 0.01 * volume
        loss = -coverage + 0.01 * volume
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # box volume
        print('volume: ', volume.item())
        print('coverage: ', coverage.item())

        # plot coverage (volume) dependence
        plt.plot(volume.item() / volume0, coverage.item(), 'b.')

        if vis:
            with torch.no_grad():
                viewer.clear_geometries()
                pcd = visualize_points3D(points, point_weights, vis=False)
                viewer.add_geometry(pcd)
                line_set = visualize_bbox(box_size.cpu().numpy(), box_pose.cpu().numpy(), vis=False)
                viewer.add_geometry(line_set)
                line_set0 = visualize_bbox(lwh0, pose, vis=False, color=(0, 0, 1))
                viewer.add_geometry(line_set0)
                viewer.run()
            # return

    if vis:
        with torch.no_grad():
            viewer.destroy_window()
            pcd = visualize_points3D(points, point_weights, vis=False)
            line_set = visualize_bbox(box_size.cpu().numpy(), box_pose.cpu().numpy(), vis=False)
            line_set0 = visualize_bbox(lwh0, pose, vis=False, color=(0, 0, 1))
            o3d.visualization.draw_geometries([pcd, line_set, line_set0])

    plt.show()


def main():
    # fit_boxes_to_cloud(n_iters=100, lr=0.1, grid_res=0.2, plot_metrics=True, show=True)
    # estimate_and_optimize_box_trajectories(n_iters=100, lr=0.1, grid_res=0.5, volume_weight=0.01, data_i=0, n_frames=20)
    fit_box()


if __name__ == '__main__':
    main()
