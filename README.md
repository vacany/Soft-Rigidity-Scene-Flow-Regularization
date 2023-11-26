# Soft Rigidity Scene Flow Regularization


<!-- # Current Results on Waymo Dataset with against [FastFlow](https://github.com/Lilac-Lee/FastNSF/tree/main) -->
<!-- ![alt text](paper/exp_results/NP-vs-RNPv1.0.png) -->

# Results on StereoKITTI dataset
![alt text](paper/performance_v2.png)

# Installation
- All modules are on RCI
- Install [Fast Geodis](https://github.com/masadcv/FastGeodis) with pip install FastGeodis --no-build-isolation
- Install [PyTorch3d](https://github.com/facebookresearch/pytorch3d) with CUDA support.
- Install [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/tree/master) with CUDA support.
- Set up paths in data/PATHS.py


# DATA
- Download [Data](https://login.rci.cvut.cz/data/lidar_intensity/sceneflow/data_sceneflow.tgz) and unpack it to the folder specified in data/PATHS.py
- waymo path: /mnt/personal/vacekpa2/data/waymo/processed
- nuscenes path: /mnt/personal/vacekpa2/data/nuscenes/processed
- argoverse path: /mnt/personal/vacekpa2/data/argoverse/processed

```console
tar -xvf data_sceneflow.tgz $DATA_DIR/sceneflow
```



