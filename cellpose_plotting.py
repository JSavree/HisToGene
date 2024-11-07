from cellpose import plot

# this is for multiple images
# nimg = len(imgs)
# for idx in range(nimg):
#     maski = masks[idx]
#     flowi = flows[idx][0]
#
#     fig = plt.figure(figsize=(12,5))
#     plot.show_segmentation(fig, imgs[idx], maski, flowi, channels=channels[idx])
#     plt.tight_layout()
#     plt.show()

# Use this instead:
import numpy as np
from cellpose import plot, utils, io

dat = np.load('_seg.npy', allow_pickle=True).item()
img = io.imread('img.tif')

# plot image with masks overlaid
mask_RGB = plot.mask_overlay(img, dat['masks'],
                        colors=np.array(dat['colors']))

# plot image with outlines overlaid in red
outlines = utils.outlines_list(dat['masks'])
plt.imshow(img)
for o in outlines:
    plt.plot(o[:,0], o[:,1], color='r')

