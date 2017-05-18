import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from helper_functions import *


files = [["Car", "training_data/vehicles/GTI_Right/image0363.png"], ["Non Car", "training_data/non-vehicles/Extras/extra5752.png"]]
orient = 9
pix_per_cell = 8
cell_per_block = 2
color_space = "YCrCb"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins

for elem in files:
    plt_num = 421
    orig_image = mpimg.imread(elem[1])

    image = np.copy(orig_image)

    image = convert_color(orig_image, color_space=color_space)

    ch1 = image[:,:,0]
    ch2 = image[:,:,1]
    ch3 = image[:,:,2]


    features1, hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis = True, feature_vec=True)
    features2, hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis = True, feature_vec=True)
    features3, hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis = True, feature_vec=True)

    # Get spatial features
    spatial_features = bin_spatial(image, size=spatial_size)

    # Get color features
    hist_features = color_hist(image, nbins=hist_bins)

    fig = plt.figure(figsize=(20, 12), dpi=80)
    # fig = plt.figure(figsize=(10, 10))

    plt.subplot(plt_num)
    plt.imshow(orig_image)
    plt.title(elem[0])

    plt_num += 1
    plt.subplot(plt_num)
    plt.plot(spatial_features)
    plt.title("Spatial Features")

    plt_num += 1
    print(plt_num)
    plt.subplot(plt_num)
    plt.imshow(hog1, cmap='gray')
    plt.title("Channel 1 HOG")

    plt_num += 1
    plt.subplot(plt_num)
    plt.plot(features1)
    plt.title("Channel 1 Features")

    plt_num += 1
    plt.subplot(plt_num)
    plt.imshow(hog2, cmap='gray')
    plt.title("Channel 2 HOG")

    plt_num += 1
    plt.subplot(plt_num)
    plt.plot(features2)
    plt.title("Channel 2 Features")

    plt_num += 1
    plt.subplot(plt_num)
    plt.imshow(hog3, cmap='gray')
    plt.title("Channel 3 HOG")
    
    plt_num += 1
    plt.subplot(plt_num)
    plt.plot(features3)
    plt.title("Channel 3 Features")

    plt.tight_layout()
    plt.savefig("output_images/" + elem[0] + ".jpg")
