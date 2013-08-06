from skimage.feature import censure_keypoints
from skimage.color import rgb2gray
from skimage.data import lena
import matplotlib.pyplot as plt
img = lena()

# Initializing the parameters for Censure keypoints
gray_img = rgb2gray(img)
n_scales = 7
mode = 'DoB'
nms_threshold = 0.10
rpc_threshold = 10
kp_star, scale = censure_keypoints(gray_img, n_scales, mode, nms_threshold,
                      rpc_threshold)

f, axarr = plt.subplots(2, n_scales / 2)

# Plotting Censure features at all the scales
for i in range(n_scales - 2):
    keypoints = kp_star[scale == i + 2]
    num = len(keypoints)
    x = keypoints[:, 1]
    y = keypoints[:, 0]
    s = 5 * 2**(i + 2)
    axarr[i / 3, i - (i / 3) * 3].imshow(img)
    axarr[i / 3, i - (i / 3) * 3].scatter(x, y, s, facecolors='none',
                                          edgecolors='g')
    axarr[i / 3, i - (i / 3) * 3].set_title(' %s %s-Censure features at scale %d' % (num, mode, i + 2))

plt.suptitle('NMS threshold = %f, RPC threshold = %d' % (nms_threshold, rpc_threshold))
plt.show()
