import torch
from pytorch3d.ops.knn import knn_points


def gather_KNN_correspondences_from_flow(pc1, pc2, flow, device='cuda'):
    '''
    :param pc1: batched point cloud
    :param pc2: batch point cloud
    :param flow: flow for pc1
    :return: trajectory tensor with matched points for first time in pc1
    trajectory [BS, N, 3] where points in the row along the BS are forming the trajectory
    '''
    M = len(pc1)
    NN_indices = torch.zeros_like(pc1[..., 0], dtype=torch.long)

    for i in range(M):
        dist, NN, _ = knn_points(pc1[i:i + 1] + flow[i:i + 1], pc2[i:i + 1], K=1)

        # todo calculate with max_distance, maybe eliminate points? It should continue with trajectory trend? But that also make mistakes
        matched_indices = NN[0, ..., 0]
        NN_indices[i] = matched_indices

    trajectories = torch.zeros(pc1.shape, device=device)

    for i in range(M):
        p1 = pc1[i]
        p2 = pc2[i]
        nn1 = NN_indices[i]

        correspondences = p2[nn1]

        trajectories[i] = correspondences

    return trajectories
