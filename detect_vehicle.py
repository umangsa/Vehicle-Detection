import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from training_data_helpers import *
from helper_functions import *
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from collections import deque


TUNE_PARAMS = False
FEATURE_EXTRACTION = False
PRE_TRAINED_CLASSIFIER = True

# define the tunable parameters to extract features
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
ystart = 400
ystop = 656
ystart_stop = [[400, 500], [400, 550], [500, 656]]
xstart_stop = [[350, 1280], [250, 1280], [250, 1280]]
scales = [1.25, 1.5, 2.0]



def feature_extraction(color_space, spatial_size, hist_bins, 
    orient, pix_per_cell, 
    cell_per_block, 
    hog_channel, spatial_feat, 
    hist_feat, hog_feat):
    # Read the training file names from the training db
    cars, notcars = read_training_data()

    car_features = extract_features(cars, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # save to disk to save time in next iteration
    np.save("features/image_features", X)
    np.save("features/labels", y)

    return X, y

def process_image(image, heat_threshold = 4, png = False, debug = False):
    # return detect_cars(image, color_space = color_space, clf = clf, 
    #     X_scaler = X_scaler, orient = orient, pix_per_cell = pix_per_cell, 
    #     cell_per_block = cell_per_block, spatial_size = spatial_size, 
    #     hist_bins = hist_bins)
    hot_windows = []
    draw_image = np.copy(image)

    if png == False:
        image = image.astype(np.float32)/255

    i = 0
    for scale in scales:

        hot_windows = find_cars(image, hot_windows, color_space = color_space, 
            xstart_stop = xstart_stop[i], ystart = ystart_stop[i][0], ystop = ystart_stop[i][1], 
            scale = scale, svc = clf, X_scaler = X_scaler, orient = orient, 
            pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, spatial_size = spatial_size, 
            hist_bins = hist_bins)
        i += 1

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows) 

    heatmaps.append(heat)
    heatmap_sum = sum(heatmaps)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heatmap_sum, heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    detections =  draw_labeled_bboxes(draw_image, labels)

    if debug:
        plt.subplot(232)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6) 
        plt.imshow(window_img)
        plt.title("Hot boxes")

        plt.subplot(233)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')

        plt.subplot(234)
        plt.imshow(labels[0], cmap='gray')
        title = str(labels[1]) + ' cars found'
        plt.title(title)

        plt.subplot(235)
        plt.imshow(detections)
        plt.title("Detected Cars")

    return detections



if FEATURE_EXTRACTION:
    print("Extracting features")
    t = time.time()
    X, y = feature_extraction(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print("Feature extraction time ", round(t2-t, 2))
else:
    print("Loading features")
    X = np.load("features/image_features.npy")
    y = np.load("features/labels.npy")

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)


# Split up data into randomized training and test sets
print("Splitting training and validation data")
rand_state = np.random.randint(0, 100)
X_train, X_valid, y_train, y_valid = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# splitter = TimeSeriesSplit(n_splits=2)
# X_train, X_valid  = splitter.split(scaled_X)
# y_train, y_valid  = splitter.split(y)

if PRE_TRAINED_CLASSIFIER:
    clf = joblib.load('features/classifier.pkl')
    print("Classifier loaded")
else:
    if TUNE_PARAMS:
        # initialize the SVM for training on the data
        # Parameter tuning applied to the SVC
        # parameters = {'C':[0.01, 0.1, 1, 10, 100]}
        parameters = { 'C':[0.0001, 0.001, 0.01]}
        # Use a linear SVC 
        # svc = LinearSVC()
        svc = LinearSVC()
        clf = GridSearchCV(svc, parameters)

        # Check the training time for the SVC
        print("Starting SVC Fit")
        t=time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        print("Best parameters for training = {}".format(clf.best_params_))
    else:
        clf = LinearSVC( C=0.0001)
        # Check the training time for the SVC
        print("Starting SVC Fit")
        t=time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')

    joblib.dump(clf, 'features/classifier.pkl') 
    print("Saved trained model to features/classifier.pkl")  

    print("Checking accuracy of trained model")
    print('Validation Accuracy of SVC = ', round(clf.score(X_valid, y_valid), 4))

heatmaps = deque(maxlen=1)

# get size of images
# assumption is that all images are of same size
i = 1
for filename in glob.iglob('test_images/*.jpg'):
    image = mpimg.imread(filename)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    fig = plt.figure(figsize=(16, 9))
    plt.subplot(231)
    plt.imshow(image)
    plt.title(filename)


    t=time.time()
    window_img = process_image(image, heat_threshold = 1, png = False, debug = True)
    t2 = time.time()
    print("Detection time = {}".format(t2-t))


    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

    # plt.imshow(window_img)
    plt.savefig("output_images/test" + str(i) + ".jpg")
    i += 1


heatmaps = deque(maxlen=10)
heatmaps.clear()
output = 'output_images/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)
