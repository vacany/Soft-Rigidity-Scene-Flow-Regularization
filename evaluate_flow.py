import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.PATHS import EXP_PATH

# from vis.deprecated_vis import *


from models.NP import PoseNeuralPrior

from loss.flow import SC2_KNN, GeneralLoss, MBSC

from ops.metric import SceneFlowMetric
from data.dataloader import NSF_dataset

from configs.evalute_datasets import cfg as cfg_df
from models.seed import seed_everything

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    # SEED = 42
    # seed_everything(SEED)
    # print(SEED)
    print(cfg_df)
    
    # Reformat current config
    cfg_int = int(sys.argv[1])
    cfg = cfg_df.iloc[cfg_int].to_dict()
 
    
    # Exp path is defined in config files (PATHS.py)
    exp_folder = EXP_PATH + f'/ours_kitti_t/{cfg_int}'
    cfg['exp_folder'] = exp_folder

    for fold in ['inference', 'visuals']:
        os.makedirs(exp_folder + '/' + fold, exist_ok=True)

    for run in range(cfg['runs']):

        metric = SceneFlowMetric()
        dataset = NSF_dataset(dataset_type=cfg['dataset_type'])

        for f, data in enumerate(tqdm(dataset)):
            
            max_loss = 100000
            pc1 = data['pc1'].to(device)
            pc2 = data['pc2'].to(device)
            gt_flow = data['gt_flow'].to(device)

            # Init model for each sequence frame
            if cfg['model'] == 'NP':
                model = PoseNeuralPrior(pc1, pc2, init_transform=cfg['init_transform'], use_transform=cfg['use_transform'],
                                eps=cfg['eps'], min_samples=cfg['min_samples']).to(device)
            
            elif cfg['model'] == 'SCOOP':
                from models.scoopy.get_model import PretrainedSCOOP
                model = PretrainedSCOOP().to(device)
                model.update(pc1, pc2)


            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

            # Parametrized sceneflow loss module
            LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=cfg['K'], max_radius=cfg['max_radius'],
                                     smooth_weight=cfg['smooth_weight'],
                                     forward_weight=cfg['forward_weight'], sm_normals_K=cfg['sm_normals_K'], pc2_smooth=True)


            st = time.time()
            if cfg['SC2'] == 'MBSC':
                SC2_Loss = MBSC(pc1, eps=cfg['eps'], min_samples=cfg['min_samples'])
            if cfg['SC2'] == 'SC2_KNN':
                SC2_Loss = SC2_KNN(pc1=pc1, K=cfg['K'], use_normals=cfg['use_normals'])


            for flow_e in range(cfg['max_iters']):
                pc1 = pc1.contiguous()

                pred_flow = model(pc1)


                loss = LossModule(pc1, pred_flow, pc2)

                if cfg['beta'] > 0:
                    loss += cfg['beta'] * SC2_Loss(pred_flow)

                loss.mean().backward()

                optimizer.step()
                optimizer.zero_grad()


            data['eval_time'] = time.time() - st

            data['pred_flow'] = pred_flow.detach().cpu()
            metric.update(data)

            if run == 0:
                np.savez(exp_folder + f'/inference/sample-{f}.npz',
                         pc1=pc1.detach().cpu().numpy(),
                         pc2=pc2.detach().cpu().numpy(),
                         pred_flow=pred_flow.detach().cpu().numpy(),
                         gt_flow=gt_flow.detach().cpu().numpy(),
                         )
            # break

        # Experiment description for current info
        print_str = ''
        print_str += 'EXPS for SC2'
        print(print_str)
        print(metric.get_metric().mean())

        # Store metrics
        metric.store_metric(exp_folder + f'/metric-run-{run}.csv')
        pd.DataFrame(cfg, index=[0]).to_csv(exp_folder + '/config.csv')