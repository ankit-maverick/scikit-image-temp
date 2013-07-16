import numpy as np
from skimage import data
from skimage import transform as tf
from skimage.feature import pairwise_hamming_distance, brief, match_keypoints_brief, corner_harris, corner_peaks
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt

rotate = 0.05
translate = (15, 20)

img_color = data.lena()
tform = tf.SimilarityTransform(scale = 1, rotation=rotate, translation=translate)
transformed_img_color = tf.warp(img_color, tform)
img = rgb2gray(img_color)
transformed_img = rgb2gray(transformed_img_color)

# Extracting keypoints using Harris corner response and describing them
# using BRIEF for both the images
keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
descriptors1, keypoints1 = brief(img, keypoints1, descriptor_size=512)

keypoints2 = corner_peaks(corner_harris(transformed_img), min_distance=5)
descriptors2, keypoints2 = brief(transformed_img, keypoints2, descriptor_size=512)

# Matching the BRIEF described keypoints in both the images using
# Hamming distance dissimilarity measure
pairwise_hamming_distance(descriptors1, descriptors2)
matched_keypoints = match_keypoints_brief(keypoints1, descriptors1, keypoints2, descriptors2, threshold=0.15)

print "Pairs of matched keypoints :\n"
print matched_keypoints

# Plotting the matched correspondences in both the images using matplotlib
src = matched_keypoints[:, 0, :]
dst = matched_keypoints[:, 1, :]

img_combined = np.concatenate((img_as_float(img_color), transformed_img_color), axis=1)
offset = img.shape

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.gray()

ax.imshow(img_combined, interpolation='nearest')
ax.axis('off')
ax.axis((0, 2 * offset[1], offset[0], 0))
ax.set_title('Matched correspondences : Rotation = %f; Translation = %s; threshold = 0.15' % (rotate, translate,))

for m in range(len(src)):
	ax.plot((src[m, 1], dst[m, 1] + offset[0]), (src[m, 0], dst[m, 0]), '-', color='g')
	ax.plot(src[m, 1], src[m, 0], '.', markersize=10, color='g')
	ax.plot(dst[m, 1] + offset[0], dst[m, 0], '.', markersize=10, color='g')

plt.show()
