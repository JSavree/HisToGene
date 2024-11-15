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
from deepcell.utils.plot_utils import make_outline_overlay 677 b nnm,nnm

img_data = img[:,:,[2,1]]
img_data = np.reshape(img_data, (1, img_data.shape[0], img_data.shape[1], 2))
rgb_image = create_rgb_image(img_data, channel_colors=['blue', 'green'])
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.imshow(make_outline_overlay(rgb_image, predictions=np.reshape(mask, (1, mask.shape[0], mask.shape[1], 1)))[0])
fig.savefig('')


