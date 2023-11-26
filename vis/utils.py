import numpy as np
import torch
# import pptk
import open3d as o3d

def visualize_points3D(points, labels=None, point_size=0.02, **kwargs):
    # if not socket.gethostname().startswith("Pat"):
    #     return

    if type(points) is not np.ndarray:
        points = points.detach().cpu().numpy()

    if type(labels) is not np.ndarray and labels is not None:
        labels = labels.detach().cpu().numpy()


    if labels is None:
        v = pptk.viewer(points[:,:3])
    else:
        v = pptk.viewer(points[:, :3], labels)

    v.set(point_size=point_size)
    v.set(**kwargs)

    return v


def visualize_multiple_pcls(*args, **kwargs):
    p = []
    l = []

    for n, points in enumerate(args):
        if type(points) == torch.Tensor:
            p.append(points[:,:3].detach().cpu().numpy())
        else:
            p.append(points[:,:3])
        l.append(n * np.ones((points.shape[0])))

    p = np.concatenate(p)
    l = np.concatenate(l)
    v=visualize_points3D(p, l)
    v.set(**kwargs)

def visualize_plane_with_points(points, n_vector, d):

    xx, yy = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), 100),
                         np.linspace(points[:,1].min(), points[:,1].max(), 100))

    z = (- n_vector[0] * xx - n_vector[1] * yy - d) * 1. / n_vector[2]
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    z = np.concatenate(z)

    plane_pts = np.stack((x, y, z, np.zeros(z.shape[0]))).T

    d_dash = - n_vector.T @ points[:,:3].T

    bin_points = np.concatenate((points, (d - d_dash)[:, None]), axis=1)

    vis_pts = np.concatenate((bin_points, plane_pts))

    visualize_points3D(vis_pts, vis_pts[:,3])


def visualize_flow3d(pts1, pts2, frame_flow):
    # flow from multiple pcl vis
    # valid_flow = frame_flow[:, 3] == 1
    # vis_flow = frame_flow[valid_flow]
    # threshold for dynamic is flow larger than 0.05 m

    if len(pts1.shape) == 3:
        pts1 = pts1[0]

    if len(pts2.shape) == 3:
        pts2 = pts2[0]

    if len(frame_flow.shape) == 3:
        frame_flow = frame_flow[0]

    if type(pts1) is not np.ndarray:
        pts1 = pts1.detach().cpu().numpy()

    if type(pts2) is not np.ndarray:
        pts2 = pts2.detach().cpu().numpy()

    if type(frame_flow) is not np.ndarray:
        frame_flow = frame_flow.detach().cpu().numpy()

    dist_mask = np.sqrt((frame_flow[:,:3] ** 2).sum(1)) > 0.05

    vis_pts = pts1[dist_mask,:3]
    vis_flow = frame_flow[dist_mask]


    all_rays = []

    for x in range(1, int(20)):
        ray_points = vis_pts + (vis_flow[:, :3]) * (x / int(20))
        all_rays.append(ray_points)

    all_rays = np.concatenate(all_rays)

    visualize_multiple_pcls(*[pts1, all_rays, pts2], point_size=0.02, show_grid=False, lookat=[0,0,0])


if __name__ == '__main__':
    pc1 = torch.rand(1, 100, 3)
    pc2 = pc1 + 2
    flow = torch.rand(1, 100, 3)
    gt_flow = pc2 - pc1

    visualize_flow3d(pc1[0], pc2[0], flow[0])
    visualize_flow3d(pc1[0], pc2[0], gt_flow[0])

    if vis:
        fig, ax = plt.subplots(3, figsize=(10, 10), dpi=200)

        ax[0].set_title('Flow')
        ax[0].axis('equal')
        ax[0].plot(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(), 'b.', alpha=0.7,
                   markersize=1)
        ax[0].plot(pc2[0, :, 0].detach().cpu().numpy(), pc2[0, :, 1].detach().cpu().numpy(), 'r.', alpha=0.7,
                   markersize=1)
        # ax[0].quiver(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(), gt_flow[0, :, 0].detach().cpu().numpy(), gt_flow[0, :, 1].detach().cpu().numpy(), color='g', width=0.001, angles = 'xy', scale_units = 'xy', scale = 1)
        ax[0].plot((pc1 + pred_flow)[0, :, 0].detach().cpu().numpy(),
                   (pc1 + pred_flow)[0, :, 1].detach().cpu().numpy(), 'g.', alpha=0.5, markersize=1)

        ax[1].set_title('EPE')
        ax[1].axis('equal')
        epe = torch.norm(gt_flow[:, :, :3] - pred_flow, dim=-1)
        ax[1].scatter(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(),
                      c=epe[0].detach().cpu().numpy(), s=1, cmap='jet')
        ax[2].plot(epe[0].detach().cpu().numpy(), 'b*', alpha=0.7, markersize=2)

        cb = fig.colorbar(ax[1].collections[0], ax=ax[1])
        cb.set_label('End-point-Error')

        instance_classes = torch.argmax(mask, dim=2)[0].detach().cpu().numpy()

        data_save = {'pred_flow': pred_flow.detach().cpu().numpy(), 'id_mask1': instance_classes,
                     't': t.detach().cpu().numpy(), 'yaw': yaw.detach().cpu().numpy()}
        np.savez(exp_folder + f'/inference/{f:04d}.npz', **data_save)
        fig.savefig(exp_folder + f'/visuals/{f:04d}.png')
        plt.close()
    # # visualize ground truth flow with open3d
    # pc1 = pc1[0].detach().cpu().numpy()
    # pc2 = pc2[0].detach().cpu().numpy()
    #
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pc1)
    #
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(pc2)
    #
    # # visualize ground truth flow with open3d as arrows from pc1 to pc2
    # pcd1.normals = o3d.utility.Vector3dVector(gt_flow[0].detach().cpu().numpy())
    #
    # o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=True)
