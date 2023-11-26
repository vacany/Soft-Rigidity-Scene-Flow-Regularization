# metric DO NOT delete
# import glob
# import numpy as np
# import pandas as pd
# exp_folder = f'/mnt/personal/vacekpa2/experiments/' + 'development_metric'
# all_paths = sorted(glob.glob(exp_folder + '/*'))
#
# for path in all_paths:
#
#     metric_df = pd.read_csv(path + '/metric.csv', index_col=0)
#     cfg_df = pd.read_csv('tse.csv', index_col=0).iloc[0]
import os

import torch
from tqdm import tqdm
import pandas as pd
import time

from models.RSNF import NeuralPriorNetwork
from loss.flow import GeneralLoss, chamfer_distance_loss
from models.scoopy.get_model import PretrainedSCOOP
from ops.metric import SceneFlowMetric
from data.dataloader import SFDataset4D, NSF_dataset
from vis.deprecated_vis import plt, visualize_flow3d

# cfg
plt.close()
device = torch.device('cuda:0')


# cfg
smooth_weight = [0, 0.1, 0.5, 1, 3, 5, 10, 15]
forward_weight = [0, 0.1, 0.5, 1, 3, 5, 10, 15]

cfg_list = []

for smooth in smooth_weight:
    for forward in forward_weight:
        cfg_list.append({'smooth_weight': smooth, 'forward_weight': forward})

# for smooth in smooth_weight:
#     cfg_list.append({'smooth_weight': smooth, 'forward_weight': 10})
#
# for forward in forward_weight:
#     cfg_list.append({'smooth_weight': 10, 'forward_weight': forward})


dataset_type = 'kitti_t'
metric_list = []

if dataset_type.startswith('kitti'):
    net = PretrainedSCOOP().to(device)
    lr = 0.02

else:
    net = NeuralPriorNetwork(lr=0.008).to(device)
    lr = 0.008

for _, cfg in enumerate(tqdm(cfg_list)):

    dataset = NSF_dataset(dataset_type=dataset_type)
    alpha_surf = cfg['smooth_weight']
    alpha_csm = cfg['forward_weight']

    metric = SceneFlowMetric()

    for idx, data in enumerate(dataset):

        pc1 = data['pc1'][:1].to(device)
        pc2 = data['pc2'][:1].to(device)
        data['gt_flow'] = data['gt_flow'][:1].to(device)


        st = time.time()
        LossModule = GeneralLoss(pc1, pc2, K=8, sm_normals_K=4, smooth_weight=alpha_surf, forward_weight=alpha_csm, pc2_smooth=True)
        net.update(pc1, pc2)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for i in range(150):
            pred_flow = net(pc1)
            data['pred_flow'] = pred_flow

            loss = LossModule(pc1, pred_flow, pc2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(i, loss)
            # criterion of convergence


        data['eval_time'] = time.time() - st

        metric.update(data)  # per one data sample

        # if idx == 3:
        #     break

    metric_list.append(metric)

    # if _ == 2:
    #     break

os.makedirs(f'/mnt/personal/vacekpa2/experiments/pcflow/{dataset_type}_ablation_weights', exist_ok=True)

# final print
final_list = []
for i in range(len(metric_list)):
    df =  metric_list[i].get_metric().mean()

    epe = df['EPE']
    accs = df['AS']
    accr = df['AR']
    eval_time = df['Eval_Time']
    print(cfg_list[i], "EPE: ", f"{epe:4f}", "AS: ", f"{accs:.4f}", "AR: ", f"{accr:.4f}, Eval Time: ", f"{eval_time:.4f}")

    final_list.append([cfg_list[i]['smooth_weight'], cfg_list[i]['forward_weight'], epe, accs, accr, eval_time])

final_df = pd.DataFrame(final_list, columns=['alpha_surf', 'alpha_csm', 'EPE', 'AS', 'AR', 'Eval_Time'])
final_df.to_csv(f'/mnt/personal/vacekpa2/experiments/pcflow/{dataset_type}_ablation_weights/ablation_weights.csv')

fig, ax = plt.subplots(2)
ax[0].plot(final_df['alpha_surf'], final_df['AS'], 'g.', label='AS')
ax[0].set_title('Alpha Surf')
ax[0].grid()
ax[1].plot(final_df['alpha_csm'], final_df['AS'], 'b.', label='AS')
ax[1].set_title('Alpha CSM')
ax[1].grid()

fig.savefig(f'/mnt/personal/vacekpa2/experiments/pcflow/{dataset_type}_ablation_weights/ablation_weights.png')
