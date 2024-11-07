import sys
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from cellotype.predict import CelloTypePredictor

img = io.imread('')

model = CelloTypePredictor('models/tissuenet_model_0019999.pth',confidence_thresh=0.3, max_det=1000, device='cuda',
                           config_path='configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml')

mask = model.predict(img)

from deepcell.utils.plot_utils import create_rgb_image
from deepcell.utils.plot_utils import make_outline_overlay



