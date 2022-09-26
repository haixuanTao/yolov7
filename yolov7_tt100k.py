import os
import sys
from os.path import exists

import gdown

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# from demo import visualization_ours
parent_path = os.path.dirname(__file__)
WEIGHTS = os.path.join(parent_path, "best.pt")


if not exists(WEIGHTS):
    url = "https://drive.google.com/uc?export=download&id=1SdPdUMzEPQLdIjXqFgiL29GsWhqgGj95"
    gdown.download(url, WEIGHTS)
