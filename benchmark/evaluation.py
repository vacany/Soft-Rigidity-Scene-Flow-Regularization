# Imports
import sys
import time
sys.path.insert(0, '/home/vacekpa2/4D-RNSFP')

from tqdm import tqdm

from ops.metric import SceneFlowMetric
from data.dataloader import SFDataset4D
# from vis.deprecated_vis import imshow, visualize_flow3d

from models.RSNF import *  # RigidNeuralPrior, NeuralPrior, FreespaceRigidNeuralPrior

def get_datetime():
    return "-".join([str(time.localtime()[i]) for i in range(0,6)])
def run_sceneflow_experiment(exp_config):

    for run in range(exp_config['nbr_of_runs']):
        SF_metric = SceneFlowMetric()
        dataset = SFDataset4D(**exp_config)

        # builder?
        model = NeuralPrior(lr=0.008, early_stop=5, loss_diff=0.001, dim_x=3, filter_size=128, layer_size=8).to(device)
        # model = RigidNeuralPrior(lr=0.008, early_stop=5, loss_diff=0.001, dim_x=3, filter_size=128, layer_size=8).to(device)
        # model = FreespaceRigidNeuralPrior(lr=0.008, early_stop=10, loss_diff=0.001, dim_x=3, filter_size=128, layer_size=8).to(device)

        print("Processing: ", exp_config['dataset_type'].capitalize())
        print('With Model: ', model._get_name())

        for frame_id in tqdm(range(len(dataset))):
            data = dataset.__getitem__(frame_id)

            data['pc1'] = data['pc1'].to(device)
            data['pc2'] = data['pc2'].to(device)
            data['gt_flow'] = data['gt_flow'].to(device)

            model.initialize()

            start_time = time()
            data = model(data)
            end_time = time()
            data['eval_time'] = end_time - start_time

            SF_metric.update(data)

        print(SF_metric.get_metric().mean())

if __name__ == '__main__':


    pass
