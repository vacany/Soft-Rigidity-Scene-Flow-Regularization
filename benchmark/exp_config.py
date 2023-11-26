exp_list = []
version = 1.0
# path_name_by = 'Model'



for dataset in ['argoverse', 'nuscenes', 'waymo', 'kitti_t']:

    for model_name in ['NeuralPrior', 'RigidNeuralPriorV2']:
        exp_config = {
        'version' : version,
        'model_name' : model_name,
        'dataset_type' : dataset,
        'data_split' : 'train*',
        'sequence' : '*',
        'n_frames' : 1,
        'only_first' : True,
        }

        exp_list.append(exp_config)

# Nuscenes_exp = {
# 'version' : version,
# 'model_name' : 'dev',
# 'dataset_type' : 'nuscenes',
# 'data_split' : 'train',
# 'sequence' : 'scene-*',
# 'n_frames' : 1,
# 'only_first' : True,
# }
#
# # todo how to keep configs, dynamically setup? One structure and keep it. The only important thing is to learn to use it fast
# Waymo_exp = {
# 'version' : version,
# # 'model_name' : 'RigidNeuralPriorV2',
# 'model_name' : 'dev',
# 'dataset_type' : 'waymo',
# 'data_split' : 'train',
# # 'sequence' : 'segment-9653249092275997647_980_000_1000_000_with_camera_labels.tfrecord',
# 'sequence' : '*',
# 'n_frames' : 1,
# 'only_first' : True,
# }


