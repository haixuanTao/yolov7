import os
import zipfile
from os.path import exists

import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# from demo import visualization_ours
parent_path = os.path.dirname(__file__)
WEIGHTS = os.path.join(parent_path, "runs/train/yolov7-custom17/weights/best.pt")
