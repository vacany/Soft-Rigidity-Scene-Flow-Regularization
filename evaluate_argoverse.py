import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix
from sklearn.cluster import DBSCAN
from data.PATHS import DATA_PATH, EXP_PATH
from loss.flow import *

from vis.deprecated_vis import *
from models.simple_models import InstanceSegModule, FlowSegPrior, Weights_model, JointFlowInst
import matplotlib.pyplot as plt

from loss.flow import chamfer_distance_loss, GeneralLoss, FastNN
from loss.instance import DynamicLoss, InstanceSmoothnessLoss
from loss.utils import find_robust_weighted_rigid_alignment
from models.RSNF import NeuralPriorNetwork, RigidMovementPriorNetwork
from loss.flow import DT

from torch_scatter import scatter
from ops.metric import SceneFlowMetric
from data.dataloader import NSF_dataset


class JointModel(torch.nn.Module):

    def __init__(self, pc1, eps=0.5, min_samples=5, instances=10, init_cluster=None):
        super().__init__()
        self.instances = instances

        if init_cluster:
            # print('clustering')
            # oversegmentation
            clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pc1[0].detach().cpu().numpy()) + 1

            instances = clusters.max()
            # split to mask
            mask = torch.zeros((1, pc1.shape[1], instances), device=device)
            for i in range(instances):  # can be speed up
                mask[0, clusters == i, i] = 1
            self.mask = torch.nn.Parameter(mask, requires_grad=True)
            # visualize_points3D(pc1[0], mask.argmax(dim=2)[0])
        else:
            self.mask = torch.nn.Parameter(torch.randn(1, pc1.shape[1], instances, requires_grad=True))

        self.flow_net = RigidMovementPriorNetwork()

        # todo add boxes here?

    def forward(self, pc1, pc2=None):

        output = self.flow_net(pc1)
        mask = self.mask.softmax(dim=2)

        # Assign rigid parameters
        t = output[:, :, :3]
        yaw = output[:, :, 3:4]

        # construct rotation
        full_rotvec = torch.cat((torch.zeros((1, yaw.shape[1], 2), device=device), yaw), dim=-1)
        rot_mat = axis_angle_to_matrix(full_rotvec.squeeze(0))  # this is lot, but still okay

        # construct centroids
        ind = torch.argmax(mask, dim=2).squeeze(0)
        v_max = scatter(pc1[0], ind, dim=0, reduce='max')
        v_min = scatter(pc1[0], ind, dim=0, reduce='min')
        x_c = (v_max + v_min) / 2

        # shift points
        point_x_c = x_c[ind]

        deformed_pc = (rot_mat @ (pc1 - point_x_c).permute(1, 2, 0)).permute(2, 0, 1) + point_x_c + t

        pred_flow = deformed_pc - pc1

        return pred_flow, mask, t, yaw


def generate_configs():
    import itertools
    permutations = {'dataset_type': ['argoverse'],
                    'lr': [0.001, 0.008],
                    'K': [4, 8, 16],
                    'eps': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    'min_samples': [5, 10],
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [1,3],
                    'forward_weight': [0],
                    'sm_normals_K': [4],
                    'init_cluster': [True],
                    'early_patience': [100],
                    'max_iters': [500,1000],
                    'exp_name' : ['multi-rigid-argoverse'],
                    'runs': [5]}

    combinations = list(itertools.product(*permutations.values()))
    df = pd.DataFrame(combinations, columns=permutations.keys())

    return df

# def store_experiment():
#     pass

if __name__ == '__main__':
    # if torch.cuda.is_available():

    # else:
    #     device = torch.device('cpu')

    # cfg = {'dataset_type' : 'argoverse', 'lr': 0.008, 'K': 8, 'max_radius': 1, 'grid_factor': 10, 'smooth_weight': 3, 'forward_weight': 0, 'sm_normals_K': 4, 'init_cluster': True,
    #        'early_patience' : 30, 'max_iters' : 150, 'runs' : 1}
    cfg_df = generate_configs()

    cfg_int = int(sys.argv[1])
    print('Number of Experiments to run', len(cfg_df))
    print("Running experiment: ", cfg_int)

    cfg = cfg_df.iloc[cfg_int].to_dict()

    print(cfg)

    exp_folder = EXP_PATH + f'/{cfg["exp_name"]}/{cfg_int}'
    cfg['exp_folder'] = exp_folder


    device = torch.device(0)

    for fold in ['inference', 'visuals']:
        os.makedirs(exp_folder + '/' + fold, exist_ok=True)

    for run in range(cfg['runs']):

        metric = SceneFlowMetric()
        dataset = NSF_dataset(dataset_type=cfg['dataset_type'])

        for f, data in enumerate(tqdm(dataset)):
            # *_, data = dataset
            # if f != 26:
            #     continue

            pc1 = data['pc1'].to(device)
            pc2 = data['pc2'].to(device)
            gt_flow = data['gt_flow'].to(device)

            st = time.time()
            model = JointModel(pc1, eps=cfg['eps'], min_samples=cfg['min_samples'], instances=30, init_cluster=cfg['init_cluster']).to(device)

            # KITTI_T best lr 0.001

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            # losses
            # sm_normals_K = 4 for sota
            LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=cfg['K'], max_radius=cfg['max_radius'], smooth_weight=cfg['smooth_weight'],
                                     forward_weight=0, sm_normals_K=cfg['sm_normals_K'], pc2_smooth=True)

            for flow_e in range(cfg['max_iters']):
                pc1 = pc1.contiguous()
                pred_flow, mask, t, yaw = model(pc1)  # Model outputs directly mask and predicted flow
                # todo backward flow from two DTs
                # for mask prob
                # mask_probs = mask.max(dim=2)[0]
                # forw_dist, forward_nn, _ = knn_points(pc1 + pred_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)
                # back_dist, backward_nn, _ = knn_points(pc2, pc1 + pred_flow, lengths1=None, lengths2=None, K=1, norm=1)
                # dist_loss = ((forw_dist).mean() + (back_dist).mean()) / 2

                # dist_loss = ((forw_dist * mask_probs).mean() + (back_dist).mean()) / 2

                data['pred_flow'] = pred_flow

                loss = LossModule(pc1, pred_flow, pc2)
                inst_smooth, _ = LossModule.smoothness_loss(mask, LossModule.NN_pc1)  # added instances
                trans_smooth, _ = LossModule.smoothness_loss(t, LossModule.NN_pc1)  # added instances
                yaw_smooth, _ = LossModule.smoothness_loss(yaw, LossModule.NN_pc1)  # added instances

                # regularizations
                # loss += inst_smooth.mean() + trans_smooth.mean() + yaw_smooth.mean() + (yaw ** 2).mean() #+ (pred_flow.norm(dim=-1)).mean() * 10

                loss.mean().backward()

                optimizer.step()
                optimizer.zero_grad()

                # print(f"Iter: {flow_e:03d} \t Loss: {loss.mean().item():.4f} \t Flow: {loss.mean().item():.4f} "
                #       f"\t Smoothness: {inst_smooth.mean().item():.4f} "
                # f"Cycle: {cycle_smooth_flow.mean().item():.4f} \t Dynamic: {dynamic_loss.mean().item():.4f} \t"
                # f"RMSD: {rmsd.mean().item():.4f} \t Kabsch_w: {kabsch_w.mean().item():.4f}"
                # )


            # Update metric
            data['eval_time'] = time.time() - st
            data['pc1'] = pc1
            data['pc2'] = pc2
            data['gt_flow'] = gt_flow
            metric.update(data)

            # store plt
            if run == 0:
                fig, ax = plt.subplots(3, figsize=(10, 10), dpi=200)

                ax[0].set_title('Flow')
                ax[0].axis('equal')
                ax[0].plot(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(), 'b.', alpha=0.7, markersize=1)
                ax[0].plot(pc2[0, :, 0].detach().cpu().numpy(), pc2[0, :, 1].detach().cpu().numpy(), 'r.', alpha=0.7,  markersize=1)
                # ax[0].quiver(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(), gt_flow[0, :, 0].detach().cpu().numpy(), gt_flow[0, :, 1].detach().cpu().numpy(), color='g', width=0.001, angles = 'xy', scale_units = 'xy', scale = 1)
                ax[0].plot((pc1+pred_flow)[0, :, 0].detach().cpu().numpy(), (pc1+pred_flow)[0, :, 1].detach().cpu().numpy(), 'g.', alpha=0.5,  markersize=1)

                ax[1].set_title('EPE')
                ax[1].axis('equal')
                epe = torch.norm(gt_flow[:,:,:3] - pred_flow, dim=-1)
                ax[1].scatter(pc1[0, :, 0].detach().cpu().numpy(), pc1[0, :, 1].detach().cpu().numpy(), c=epe[0].detach().cpu().numpy(), s=1, cmap='jet')
                ax[2].plot(epe[0].detach().cpu().numpy(), 'b*', alpha=0.7, markersize=2)

                cb = fig.colorbar(ax[1].collections[0], ax=ax[1])
                cb.set_label('End-point-Error')

                instance_classes = torch.argmax(mask, dim=2)[0].detach().cpu().numpy()

                data_save = {'pred_flow': pred_flow.detach().cpu().numpy(), 'id_mask1': instance_classes, 't': t.detach().cpu().numpy(), 'yaw': yaw.detach().cpu().numpy()}
                np.savez(exp_folder + f'/inference/{f:04d}.npz', **data_save)
                fig.savefig(exp_folder + f'/visuals/{f:04d}.png')
                plt.close(fig)

            # if f == 0:
            #     break

        # print(metric.get_metric())
        print(metric.get_metric().mean())
        metric.store_metric(exp_folder + f'/metric-run-{run}.csv')
        pd.DataFrame(cfg, index=[0]).to_csv(exp_folder + '/config.csv')

        # visualize_flow3d(pc1[0], pc2[0], pred_flow[0])
