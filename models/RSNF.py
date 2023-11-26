# All in Torch
import time
import torch
import torch.nn as nn
from loss.flow import DT, GeneralLoss, FastNN
try:
    from lietorch import SE3
except ImportError:
    pass
from pytorch3d.transforms import euler_angles_to_matrix
from ops.transform import find_weighted_rigid_alignment
from pytorch3d.transforms import axis_angle_to_matrix
from torch_scatter import scatter
class dev(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        data['pred_flow'] = torch.zeros_like(data['pc1'], device=data['pc1'].device)
        return data

    def initialize(self):
        pass
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def construct_transform(rotation_vector: torch.Tensor, translation: torch.Tensor):
    '''
    Construct 4x4 transformation matrix from rotation vector and translation vector while perserving differentiation
    :param rotation_vector:
    :param translation:
    :return: Pose matrix
    '''
    # assert rotation_vector.shape[1:] == (3,)
    # assert translation.shape[1:] == (1, 3)

    rotation = euler_angles_to_matrix(rotation_vector, convention='XYZ')

    # r_t_matrix = torch.hstack([rotation, translation.T])

    r_t_matrix = torch.hstack([rotation, translation.unsqueeze(1)])

    one_vector = torch.zeros((len(rotation), 4, 1), device=rotation.device)
    one_vector[:, -1, -1] = 1

    pose = torch.cat([r_t_matrix, one_vector], dim=2)

    # pose = torch.vstack([r_t_matrix, torch.tensor([[0, 0, 0, 1]], device=rotation.device)])

    return pose


class PoseTransform(torch.nn.Module):
    '''
    Pose transform layer
    Works as a differentiable transformation layer to fit rigid ego-motion
    '''

    def __init__(self, BS=1, device='cpu'):
        super().__init__()
        # If not working in sequences, use LieTorch
        self.translation = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
        self.rotation_angles = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
        # self.pose = construct_transform(self.rotation_angles, self.translation).T.unsqueeze(0)

    def construct_pose(self):
        self.pose = construct_transform(self.rotation_angles, self.translation)

        return self.pose

    def forward(self, pc):
        # print(self.translation)
        pc_to_transform = torch.cat([pc, torch.ones((len(pc), pc.shape[1], 1), device=pc.device)], dim=2)

        pose = construct_transform(self.rotation_angles, self.translation)

        deformed_pc = torch.bmm(pc_to_transform, pose)[:, :, :3]

        return deformed_pc

class SE3_lietorch(torch.nn.Module):
    def __init__(self, BS=1, device='cpu'):
        super().__init__()
        self.translation = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
        self.quaternions = torch.zeros((BS, 4), device=device)
        self.quaternions[:, -1] = 1
        self.quaternions = torch.nn.Parameter(self.quaternions, requires_grad=True)

        # self.quaternions.requires_grad_(True)

    def forward(self, pc):

        lietorch_SE3 = SE3(torch.cat((self.translation, self.quaternions), dim=1))
        transform_matrix = lietorch_SE3.matrix()

        pc_to_transform = torch.cat([pc, torch.ones((len(pc), pc.shape[1], 1), device=pc.device)], dim=2)
        deformed_pc = torch.bmm(pc_to_transform, transform_matrix)[:, :, :3]

        return deformed_pc


class ModelTemplate(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model_cfg = self.store_init_params(locals())

        # self.initialize()

    def forward(self, data):
        st = time.time()

        eval_time = time.time() - st
        return data

    def model_forward(self, data):
        return data

    def initialize(self):
        self.apply(init_weights)

    def store_init_params(self, local_variables):
        cfg = {}
        for key, value in local_variables.items():
            if key not in ['self', '__class__', 'args', 'kwargs']:
                setattr(self, key, value)
                cfg[key] = value
            if key == 'kwargs':
                for k, v in value.items():
                    setattr(self, k, v)
                    cfg[k] = v
            if key == 'args':
                setattr(self, 'args', value)
                cfg[key] = value

        return cfg

class RigidMovementPriorNetwork(torch.nn.Module):
    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, initialize=True,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x + 1, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x + 1, bias=bias)))

        if initialize:
            self.apply(init_weights)

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose
        self.RigidTransform = PoseTransform()


    def update(self, pc1=None, pc2=None):
        pass
    def forward(self, pc1, pc2=None):
        deformed_pc = self.RigidTransform(pc1)
        rigid_flow = deformed_pc - pc1

        x = self.nn_layers[0](pc1)

        for layer in self.nn_layers[1:]:
            x = layer(x)

        pred_flow = x
        pred_flow[..., 1:] += rigid_flow[..., :]    # add to translation

        return pred_flow

class RigidEgoMovementPriorNetwork(torch.nn.Module):
    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, initialize=True,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x + 1, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x + 1, bias=bias)))

        if initialize:
            self.apply(init_weights)

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose
        self.RigidTransform = PoseTransform()


    def update(self, pc1=None, pc2=None):
        pass
    def forward(self, pc1, pc2=None):
        deformed_pc = self.RigidTransform(pc1)
        rigid_flow = deformed_pc - pc1

        x = self.nn_layers[0](pc1)

        for layer in self.nn_layers[1:]:
            x = layer(x)

        pred_flow = x
        pred_flow[..., 1:] += rigid_flow[..., :]

        return pred_flow
class NeuralPriorNetwork(torch.nn.Module):
    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, initialize=True,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        if initialize:
            self.apply(init_weights)

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose

    def update(self, pc1=None, pc2=None):
        pass
    def forward(self, x):

        for layer in self.nn_layers:
            x = layer(x)

        return x

class NeuralPrior(torch.nn.Module):
    '''
    Neural Prior, takes only point cloud t=1 on input and returns flow
    '''

    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, verbose=False):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose


    def forward(self, data):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """
        st = time().time()
        pc1, pc2 = data['pc1'][-1:], data['pc2'][-1:]
        DT_layer = DT(pc1, pc2)

        # x = pc1.clone()
        last_loss = torch.inf
        # Iteration of losses
        for e in range(5000):

            x = self.nn_layers[0](pc1)

            for layer in self.nn_layers[1:]:
                x = layer(x)

            pred_flow = x

            dt_loss, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)

            # truncated at two meters
            # dt_loss = per_point_dt_loss[per_point_dt_loss < 2].mean()
            dt_loss = per_point_dt_loss.mean()

            if torch.abs(last_loss - dt_loss) < self.loss_diff and e > self.early_stop:
                break
            else:
                last_loss = dt_loss

            dt_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.verbose:
                print(f"Epoch: {e:03d}, NN Loss: {dt_loss.item():.3f}")

        data['pred_flow'] = pred_flow
        data['rigid_flow'] = pred_flow.clone()
        data['eval_time'] = time.time() - st

        return data


    def initialize(self):
        self.apply(init_weights)


class GTNeuralPrior(torch.nn.Module):
    '''
    Neural Prior, takes only point cloud t=1 on input and returns flow
    '''

    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, verbose=False):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose


    def forward(self, data):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """
        pc1, pc2 = data['pc1'][-1:], data['pc2'][-1:]
        # DT_layer = DT(pc1, pc2)

        # x = pc1.clone()
        last_loss = torch.inf
        # Iteration of losses
        for e in range(1000):

            x = self.nn_layers[0](pc1)

            for layer in self.nn_layers[1:]:
                x = layer(x)

            pred_flow = x

            # dt_loss, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)

            # truncated at two meters
            # dt_loss = per_point_dt_loss[per_point_dt_loss < 2].mean()
            # dt_loss = per_point_dt_loss.mean()
            dt_loss = torch.nn.functional.mse_loss(pred_flow, data['gt_flow'][-1:, :, :3])

            if torch.abs(last_loss - dt_loss) < self.loss_diff and e > self.early_stop:
                break
            else:
                last_loss = dt_loss

            dt_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.verbose:
                print(f"Epoch: {e:03d}, NN Loss: {dt_loss.item():.3f}")

        data['pred_flow'] = pred_flow
        data['rigid_flow'] = pred_flow.clone()
        return data


    def initialize(self):
        self.apply(init_weights)

class RigidNeuralPriorV2(torch.nn.Module):
    '''
    Neural Prior with Rigid Transformation, takes only point cloud t=1 on input and returns flow and rigid flow (ego-motion flow)
    '''

    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8,
                 verbose=False):
        super().__init__()
        self.layer_size = layer_size
        self.RigidTransform = PoseTransform()

        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose

    def forward(self, data):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """
        st = time.time()
        pc1, pc2 = data['pc1'][-1:], data['pc2'][-1:]
        # Smooth_layer = SmoothnessLoss(pc1, pc2, K=4, sm_normals_K=0, smooth_weight=1, VA=False, max_radius=2, forward_weight=0, pc2_smooth=False, dataset='argoverse')
        DT_layer = DT(pc1, pc2)
        last_loss = torch.inf
        # Iteration of losses
        for e in range(150):


            deformed_pc = self.RigidTransform(pc1)
            rigid_flow = deformed_pc - pc1

            x = self.nn_layers[0](pc1)

            for layer in self.nn_layers[1:]:
                x = layer(x)

            # Sum of flows

            pred_flow = x + rigid_flow

            _, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)
            # _, per_point_rigid_loss = DT_layer.torch_bilinear_distance(pc1 + rigid_flow)

            # print(self.RigidTransform.construct_pose())
            # smooth_loss, per_point_smooth_loss = Smooth_layer.smoothness_loss(pred_flow, Smooth_layer.NN_pc1, Smooth_layer.loss_norm)

            # truncated at two meters
            dt_loss = per_point_dt_loss[per_point_dt_loss < 2].mean()
            # rigid_dt_loss = per_point_rigid_loss[per_point_rigid_loss < 2].mean()

            # Loss
            # loss = dt_loss
            # regularization = x.norm()
            loss = dt_loss #+ rigid_dt_loss  # + smooth_loss
            # (dt_loss + smooth_loss + rigid_dt_loss).backward()
            # print(e, loss)
            if torch.abs(last_loss - loss) < self.loss_diff and e > self.early_stop:
                break
            else:
                last_loss = loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # if self.verbose:
            #     print(f"Epoch: {e:03d}, NN Loss: {dt_loss.item():.3f} \t Rigid Loss: {rigid_dt_loss.item():.3f}")

        data['pred_flow'] = pred_flow
        data['rigid_flow'] = rigid_flow
        data['eval_time'] = time.time() - st
        return data

    def initialize(self):
        self.apply(init_weights)


class JointModel(torch.nn.Module):

    def __init__(self, pc1, pc2, eps=0.4, min_samples=5, instances=10, init_transform=0, use_transform=0,
                 flow_output='flow'):
        super().__init__()
        self.pc1 = pc1
        self.pc2 = pc2
        self.instances = instances
        self.flow_output = flow_output
        self.init_transform = init_transform
        self.use_transform = use_transform

        self.Trans = PoseTransform().to(self.pc1.device)
        if init_transform:
            self.initialize_transform()

        self.FlowModel = NeuralPriorNetwork()

        if flow_output == 'rigid':
            from sklearn.cluster import DBSCAN
            clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pc1[0].detach().cpu().numpy()) + 1

            instances = clusters.max()
            # split to mask
            mask = torch.zeros((1, pc1.shape[1], instances), device=self.pc1.device)
            for i in range(instances):  # can be speed up
                mask[0, clusters == i, i] = 1
            self.mask = torch.nn.Parameter(mask, requires_grad=True)

            self.FlowModel = RigidMovementPriorNetwork()


        # todo step of instances and smoothing

    def forward(self, pc1, pc2=None):

        if self.use_transform == 0:
            final_flow = self.infer_flow(pc1)

        elif self.use_transform == 1:
            rigid_flow = self.Trans(pc1) - pc1
            pred_flow = self.infer_flow(pc1)
            final_flow = rigid_flow + pred_flow

        elif self.use_transform == 2:
            deformed_pc1 = self.Trans(pc1)
            rigid_flow = deformed_pc1 - pc1
            pred_flow = self.infer_flow(deformed_pc1)
            final_flow = pred_flow + rigid_flow
        else:
            raise NotImplemented()

        return final_flow

    def infer_flow(self, pc1, pc2=None):

        if self.flow_output == 'flow':
            final_flow = self.FlowModel(pc1)

            return final_flow

        elif self.flow_output == 'rigid':

            output = self.FlowModel(pc1)
            mask = self.mask.softmax(dim=2)

            # Assign rigid parameters
            t = output[:, :, :3]
            yaw = output[:, :, 3:4]

            # construct rotation
            full_rotvec = torch.cat((torch.zeros((1, yaw.shape[1], 2), device=self.pc1.device), yaw), dim=-1)
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
            self.yaw = yaw
            self.t = t

        else:
            raise NotImplemented()

        return pred_flow

    def initialize_transform(self):

        self.Trans = PoseTransform().to(self.pc1.device)
        # Notes:
        # 1) Nechat NN init transformaci
        # 2) Init from Flow model can introduce rotation on KiTTISF making it worse
        trans_iters = 250

        if self.init_transform == 1:
            optimizer = torch.optim.Adam(self.Trans.parameters(), lr=0.03)
            TransLossModule = GeneralLoss(pc1=self.pc1, pc2=self.pc2, dist_mode='DT', K=1, max_radius=2,
                                          smooth_weight=0, forward_weight=0, sm_normals_K=0, pc2_smooth=False)

            for i in range(trans_iters):
                deformed_pc1 = self.Trans(self.pc1)
                rigid_flow = deformed_pc1 - self.pc1
                loss = TransLossModule(self.pc1, rigid_flow, self.pc2)
                # max_points = 5000

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

