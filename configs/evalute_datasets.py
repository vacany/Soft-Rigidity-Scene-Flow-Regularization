import itertools
import pandas as pd

permutations1 = {'dataset_type': ['kitti_t', 'kitti_o'],
                    'model' : ['NP'],
                    'lr': [0.001],
                    'K': [8],
                    'beta' : [1],
                    'eps': [0.8],
                    'min_samples': [30],
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [0],  # 0, 1 - use smoothness with SC2_KNN
                    'forward_weight': [0],
                    'sm_normals_K': [0],
                    'init_cluster': [False],
                    'early_patience': [0],
                    'max_iters': [500],
                    'runs': [1],
                    'use_normals': [False],  # 0, 1 - use normals for SC2 KNN search
                    'init_transform': [0],  # 0 - init as eye matrix, 1 - fit transform by NN to pc2 as init
                    'use_transform': [0],
                    'SC2': ['SC2_KNN'],  # MBSC, SC2_KNN
                    }

permutations2 = {'dataset_type': ['argoverse', 'nuscenes', 'waymo'],
                    'model' : ['NP'],
                    'lr': [0.008],
                    'K': [32],
                    'beta' : [1],
                    'eps': [0.8],
                    'min_samples': [30],
                    'max_radius': [2],
                    'grid_factor': [10],
                    'smooth_weight': [1],  # 0, 1 - use smoothness with SC2_KNN
                    'forward_weight': [0],
                    'sm_normals_K': [0],
                    'init_cluster': [False],
                    'early_patience': [0],
                    'max_iters': [500],
                    'runs': [1],
                    'use_normals': [False],  # 0, 1 - use normals for SC2 KNN search
                    'init_transform': [0],  # 0 - init as eye matrix, 1 - fit transform by NN to pc2 as init
                    'use_transform': [1],
                    'SC2': ['SC2_KNN'],  # MBSC, SC2_KNN
                    }

def generate_configs(permutations):
    ''' Create configs for parameter grid search, or dataset cross evaluation 
        Function returs pandas dataframe, where each row is one experiment config
        This structure can be easily exploited on SLURM job array sbatch'''
    
    combinations1 = list(itertools.product(*permutations.values()))


    df = pd.DataFrame(combinations1, columns=permutations.keys())

    return df

cfg1 = generate_configs(permutations=permutations1)
cfg2 = generate_configs(permutations=permutations2)

cfg = pd.concat((cfg1, cfg2))