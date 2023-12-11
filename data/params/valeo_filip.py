import glob
import numpy as np
from data.PATHS import DATA_PATH

data_path = f'{DATA_PATH}/valeo_filip/'

all_files = sorted(glob.glob(data_path + '*.npz'))[:-1]
# all_files = [f for f in all_files if not 'BASE' in f]

len_dataset = len(all_files)

train_idx = []
test_idx = [i for i in range(len_dataset) if i not in train_idx]


lidar_pose = (0, 0, 0)

fov_up= 25
fov_down = -25
H = 64 #?
W = 2048

data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'H' : H,
               'W' : W,
               }

def remap_keys(data_npz):

    data_npz = dict(data_npz)

    data_npz['gt_flow'] = - np.ones((data_npz['pc1'].shape[0],3))
    id_mask1 = - np.ones(data_npz['pc1'].shape[0])
    # data_npz['id_mask1'] = data_npz['inst_pc1']

    return data_npz

def frame_preprocess(pc1, pc2, gt_flow):
    # These are already preprocessed
    z_min = -3
    z_max = 3
    mask1 = np.logical_and(pc1[:, 2] < z_max, pc1[:, 2] > z_min)
    mask2 = np.logical_and(pc2[:, 2] < z_max, pc2[:, 2] > z_min)

    pc1 = pc1[mask1]
    pc2 = pc2[mask2]
    gt_flow = - np.ones((pc1.shape[0], 3))

    return pc1, pc2, gt_flow
