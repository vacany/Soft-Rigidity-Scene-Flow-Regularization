import os
import socket

server_name = socket.gethostname()

# Set up paths where you want to store data, visualize data, and store experiments
if server_name.startswith("boruvka"):
    DATA_PATH = f"/mnt/datagrid/personal/vacekpa2/data"
else:
    DATA_PATH = f"/mnt/personal/vacekpa2/data"
VIS_PATH = DATA_PATH + '/visuals/'
EXP_PATH = f"/mnt/personal/vacekpa2/experiments/"

# DATA_PATH = f"{os.path.expanduser('~')}/data/datasets/sceneflow/"
# VIS_PATH = f"{os.path.expanduser('~')}/CTU/sceneflow/visuals/"
# EXP_PATH = f"{os.path.expanduser('~')}/CTU/sceneflow/experiments/"

DATA_PATH = os.path.normpath(DATA_PATH)
VIS_PATH = os.path.normpath(VIS_PATH)
EXP_PATH = os.path.normpath(EXP_PATH)

for path in [DATA_PATH, VIS_PATH, EXP_PATH]:
    os.makedirs(path, exist_ok=True)

# if server_name.startswith("Pat"):
#     KITTI_SF_PATH = f"{os.path.expanduser('~')}/rci/data/kitti_sf/"
#
# # elif server_name.startswith('g') or server_name.startswith("login"):
#
# elif server_name.startswith("boruvka"):
#     KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/sceneflow/kitti_sf/"
#     TMP_VIS_PATH = f"{os.path.expanduser('~')}/pcflow/toy_samples/tmp_vis/"
#     EXP_PATH = f"{os.path.expanduser('~')}/experiments/"
#
# elif server_name.startswith("login") or server_name.startswith('g'):
#     # KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/"
#     TMP_VIS_PATH = f"{os.path.expanduser('~')}/pcflow/toy_samples/tmp_vis/"
#     EXP_PATH = f"{os.path.expanduser('~')}/experiments/"
