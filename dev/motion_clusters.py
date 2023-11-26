# Adjacent NN for motion segmentation
from data.dataloader import SFDataset4D
from pytorch3d.ops.knn import knn_points
from vis.deprecated_vis import *
from loss.flow import DT, SmoothnessLoss
import torch
import numpy as np


device = torch.device('cuda:0')

dataset = SFDataset4D(dataset_type='waymo', n_frames=5, only_first=False)
data = dataset[80]

pc1 = data['pc1'].to(device)
pc2 = data['pc2'].to(device)

# pc = pc1[:3]
pc = [pc1[i][pc1[i, :, 2] > 0.3] for i in range(0, 3)]

# pc = pc[:, pc[:,:,2] > 0.3]
c_pc = pc[1].unsqueeze(0)
b_pc = pc[2].unsqueeze(0)
f_pc = pc[0].unsqueeze(0)
# params
forth_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
back_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
optimizer = torch.optim.Adam([forth_flow, back_flow], lr=0.008)
# losses
SM_loss = SmoothnessLoss(pc1=c_pc, K=16, max_radius=1)

b_DT = DT(c_pc, b_pc)
f_DT = DT(c_pc, f_pc)

for i in range(500):
    # forth_dist, forth_nn, _ = knn_points(c_pc + forth_flow, f_pc, K=1, return_nn=True)
    # back_dist, back_nn, _ = knn_points(c_pc + back_flow, b_pc, K=1, return_nn=True)
    forth_dist, _ = f_DT.torch_bilinear_distance(c_pc + forth_flow)
    back_dist, _ = b_DT.torch_bilinear_distance(c_pc + back_flow)

    dist_loss = (forth_dist + back_dist).mean()
    smooth_loss = SM_loss(c_pc, forth_flow, f_pc) + SM_loss(c_pc, back_flow, f_pc)

    time_smooth = (forth_flow - (-back_flow)).norm(dim=2, p=1).mean()  # maybe magnitude of flow?

    loss = dist_loss + smooth_loss + time_smooth

    print(i, loss.item(), time_smooth.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# DBSCAN?
from sklearn.cluster import DBSCAN
motion_metric = 0.05
plain_NN_dist, _ = f_DT.torch_bilinear_distance(c_pc)
mos = np.logical_or((forth_flow[0].norm(dim=1, p=1) > motion_metric).detach().cpu().numpy(), plain_NN_dist.detach().cpu().numpy() > motion_metric)   # mask of static and dynamic
numpy_c_pc = c_pc.detach().cpu().numpy()
numpy_forth_flow = forth_flow.detach().cpu().numpy()

motion_pc = np.concatenate((numpy_c_pc[0, mos], numpy_forth_flow[0, mos]), axis=1)

clustering = DBSCAN(eps=0.2, min_samples=3).fit_predict(motion_pc) # id mask for dynamic only

ids = clustering

# visualize_points3D(motion_pc, clustering)
# visualize_points3D(c_pc[0], mos)

print(ids.max())



# VISUALS
# visualize_flow3d(c_pc[0], f_pc[0], back_flow[0])
# visualize_flow3d(c_pc[0], f_pc[0], forth_flow[0])
# visualize_points3D(c_pc[0], forth_flow[0].norm(dim=1, p=1) > 0.05)
# visualize_points3D(c_pc[0], forth_flow[0].norm(dim=1, p=1))
