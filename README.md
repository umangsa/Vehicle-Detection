**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.jpg
[image2]: ./output_images/NonCar.jpg
[image3]: ./output_images/test6.jpg
[image4]: ./output_images/test1.jpg
[image5]: ./output_images/test2.jpg
[image8]: ./output_images/test3.jpg
[image6]: ./output_images/test4.jpg
[image7]: ./output_images/test5.jpg
[video1]: ./output_images/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I have provided the information in this README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 112 through 121 of the file called `helper_functions.py`. The implementation of HOG feature extraction is done in the function called `get_hog_features` of the same file, line 8 

The code to extract features from all images is in `detect_vehicle.py` line 133 to 142. Implementation of the extraction steps is done in the function feature_extraction, line 41 of the same file

I started by reading in all the `vehicle` and `non-vehicle` images. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

For the Car images

![alt text][image1]

For Non Car images

![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I found that higher number of pixels_per_cell  e.g. 32 gave very little information and car detection was problematic. Setting it to 8 gave good detections of cars and reduced false positives. 

I tried playing with orientation. Between 7, 9 and 11, I found 9 gave me the best results

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the HOG, spatial and color histogram features as the feature input to the classifier. I played with SVC and LinearSVC. SVC with kernel = rbf and C = 100 was giving me accuracy ~99.5%. However, it was very slow. I also read some notes in the sklearn documentation that SVC is for small feature size. 

I then experimented with LinearSVC. I used C=0.0001 for tuning the performance. 

Code to tune the parameters is in `detect_vehicle.py`, lines 166 - 183

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with choosing random scales and overlap. I settled with scales = [1.25, 1.5, 2.0] and cells_per_step = 2. Searching for the ideal scale and cells_per_step could be done more scientifically using Naive Bayes or other ML technique. I did not attempt it though.

The sliding window search is done in the `helper_functions.py` file, lines 182-242. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)

![alt text][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also used the LinearSVC.decision_function(test_features) (`helper_functions.py` line 235 to find the probability of the prediction. I set a threshold of >= 0.1 to filter out false postivies in the box detections

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the test frames and their corresponding heatmaps:
![alt text][image3]
![alt text][image6]
![alt text][image7]
![alt text][image8]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

