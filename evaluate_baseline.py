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

from loss.flow import chamfer_distance_loss, GeneralLoss, FastNN


from models.RSNF import NeuralPriorNetwork, RigidMovementPriorNetwork, JointModel
from loss.flow import DT, MBSC

from loss.flow import SC2_KNN
from torch_scatter import scatter
from ops.metric import SceneFlowMetric
from data.dataloader import NSF_dataset
from loss.utils import find_robust_weighted_rigid_alignment
from models.MBNSF.utils import sc_utils


def generate_configs():
    import itertools
    import pandas as pd
    baseline = {'dataset_type': ['waymo', 'argoverse'],
                'lr': [0.003],
                'K': [32],
                'eps': [0.8],
                'min_samples': [30],
                'max_radius': [2],
                'grid_factor': [10],
                'smooth_weight': [0],  # 0, 1 - use smoothness with SC2_KNN
                'forward_weight': [0],
                'sm_normals_K': [0],
                'init_cluster': [False],
                'early_patience': [0],
                'max_iters': [200, 400],
                'runs': [3],
                'use_normals': [False],  # 0, 1 - use normals for SC2 KNN search
                'init_transform': [0],  # 0 - init as eye matrix, 1 - fit transform by NN to pc2 as init
                'use_transform': [0],
                # 0 - do not use trans, 1 - sum rigid and pred flow, 2 - transform is input to model
                'flow_output': ['flow'],  # flow, rigid
                'SC2': ['MBSC'],  # MBSC, SC2_KNN
                }

    combinations1 = list(itertools.product(*baseline.values()))
     # combinations2 = list(itertools.product(*permutations.values()))

    df1 = pd.DataFrame(combinations1, columns=baseline.keys())
     # df2 = pd.DataFrame(combinations2, columns=permutations.keys())

    df = df1

    return df




if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    cfg_df = generate_configs()
    cfg_df = pd.concat([cfg_df, generate_configs()], ignore_index=True)
    print(cfg_df)
    cfg_int = int(sys.argv[1])
    #
    cfg = cfg_df.iloc[cfg_int].to_dict()
    # tmp_cfg = {'dataset_type': ['kitti_t'],
    #            'lr': [0.001],
    #            'K': [32],
    #            'eps': [0.8],
    #            'min_samples': [30],
    #            'max_radius': [2],
    #            'grid_factor': [10],
    #            'smooth_weight': [0],
    #            'forward_weight': [0],
    #            'sm_normals_K': [4],
    #            'init_cluster': [True],
    #            'early_patience': [100],
    #            'max_iters': [15],
    #            'runs': [1],
    #            'use_normals' : [True],    # 0, 1 - use normals for SC2 KNN search
    #            'init_transform' : [1],    # 0 - init as eye matrix, 1 - fit transform by NN to pc2 as init
    #            'use_transform' : [1],     # 0 - do not use trans, 1 - sum rigid and pred flow, 2 - transform is input to model
    #            'flow_output' : ['flow'],  # flow, rigid
    #            'SC2' : ['SC2_KNN'],          # MBSC, SC2_KNN
    #             }

    # cfg = tmp_cfg
    # cfg = pd.DataFrame(tmp_cfg).iloc[0].to_dict()
    print(cfg)
    vis = False

    exp_folder = EXP_PATH + f'/multi-SC2-baseline-waymo/{cfg_int}'
    # exp_folder = EXP_PATH + f'/dev/'
    cfg['exp_folder'] = exp_folder

    for fold in ['inference', 'visuals']:
        os.makedirs(exp_folder + '/' + fold, exist_ok=True)

    for run in range(cfg['runs']):

        metric = SceneFlowMetric()
        dataset = NSF_dataset(dataset_type=cfg['dataset_type'])

        for f, data in enumerate(tqdm(dataset)):
            # *_, data = dataset
            # if f != 46:
            #     continue
            max_loss = 100000
            pc1 = data['pc1'].to(device)
            pc2 = data['pc2'].to(device)
            gt_flow = data['gt_flow'].to(device)


            # model = JointModel(pc1, eps=cfg['eps'], min_samples=cfg['min_samples'], instances=30, init_cluster=cfg['init_cluster']).to(device)
            # model = EgoJointModel(pc1, eps=cfg['eps'], min_samples=cfg['min_samples'], instances=30, init_cluster=cfg['init_cluster']).to(device)
            # model = NeuralPriorNetwork().to(device)

            model = JointModel(pc1, pc2, init_transform=cfg['init_transform'], use_transform=cfg['use_transform'],
                               eps=cfg['eps'], min_samples=cfg['min_samples'], flow_output=cfg['flow_output']).to(device)


            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            # losses
            # sm_normals_K = 4 for sota
            LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=cfg['K'], max_radius=cfg['max_radius'],
                                     smooth_weight=cfg['smooth_weight'],
                                     forward_weight=0, sm_normals_K=cfg['sm_normals_K'], pc2_smooth=True)


            st = time.time()
            if cfg['SC2'] == 'MBSC':
                SC2_Loss = MBSC(pc1, eps=cfg['eps'], min_samples=cfg['min_samples'])
            if cfg['SC2'] == 'SC2_KNN':
                SC2_Loss = SC2_KNN(pc1=pc1, K=cfg['K'], use_normals=cfg['use_normals'])


            for flow_e in range(cfg['max_iters']):
                pc1 = pc1.contiguous()

                pred_flow = model(pc1)

                data['pred_flow'] = pred_flow


                loss = LossModule(pc1, pred_flow, pc2)

                loss += SC2_Loss(pred_flow)
                # loss += MySC2_Loss(pred_flow)

                if cfg['flow_output'] == 'rigid':
                    # pred_flow = rigid_flow
                    trans_smooth, _ = LossModule.smoothness_loss(model.t, LossModule.NN_pc1)  # added instances
                    yaw_smooth, _ = LossModule.smoothness_loss(model.yaw, LossModule.NN_pc1)  # added instances

                # print(flow_e, loss.mean().item())
                # regularizations
                # print('using regularization')
                # loss += 10 * rigid_loss.mean() + inst_smooth.mean() + trans_smooth.mean() + yaw_smooth.mean()  #+ (pred_flow.norm(dim=-1)).mean() * 10
                # loss += inst_smooth.mean()  # + trans_smooth.mean() + yaw_smooth.mean()  #+ (pred_flow.norm(dim=-1)).mean() * 10
                # loss += (yaw ** 2).mean()
                # loss += ego_motion_loss * 0.1

                # if (flow_e <= cfg['early_patience']) or (loss.mean() < max_loss + 0.0001):
                #     max_loss = loss.mean()
                #     ulozit flow?
                # else:
                #     break

                loss.mean().backward()

                optimizer.step()
                optimizer.zero_grad()


            data['eval_time'] = time.time() - st
            data['pc1'] = pc1
            data['pc2'] = pc2
            data['gt_flow'] = gt_flow
            metric.update(data)

            if run == 0:
                np.savez(exp_folder + f'/inference/sample-{f}.npz',
                         # pc1=pc1.detach().cpu().numpy(),
                         # pc2=pc2.detach().cpu().numpy(),
                         pred_flow=pred_flow.detach().cpu().numpy(),
                         # gt_flow=gt_flow.detach().cpu().numpy(),
                         )


        # print(metric.get_metric())
        print_str = ''
        print_str += 'EXPS for SC2'
        print(print_str)
        print(metric.get_metric().mean())

        metric.store_metric(exp_folder + f'/metric-run-{run}.csv')
        pd.DataFrame(cfg, index=[0]).to_csv(exp_folder + '/config.csv')

        # visualize_flow3d(pc1[0], pc2[0], pred_flow[0])


# todo this might be the way, ego-transform using SC2 and KNN SC2

