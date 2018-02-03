
# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane.png "Center lane driving"
[image2]: ./examples/recovery_1.png "Recovery Image"
[image3]: ./examples/recovery_2.png "Recovery Image"
[image4]: ./examples/recovery_3.png "Recovery Image"
[image5]: ./examples/normal.png "Normal Image"
[image6]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 164-173) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 163). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 168 and 175). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 189).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used around 5700 rows of simulator data to train this network. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use architecture which is proven to be good at deriving steering angles from images. This model is similar to which NVIDIA used here https://devblogs.nvidia.com/deep-learning-self-driving-cars/ but it is not 100% the same. I used 500 elements in first fully connected layer not 1164. 
 I thought this model might be appropriate because it shows good results based on data from NVIDIA blog and is robust.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it uses dropout layer.

Then I rerun the simulator to gather better data. I added more data for recovery driving from the sides and smooth driving around curves.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, for example at the first curve after the bridge and at the second curve after the bridge. To improve the driving behavior in these cases, I've generated more data with better recoveray from the sides and trained one more epoch with those data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Cropping         		| Crop top 50 pixels and bottom 20				|
| Lambda         		| Normalization									|
| Convolution 5x5     	| 2x2 stride, same padding, outputs 45x160x24	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, same padding, outputs 23x80x36	|
| RELU					|												|
| Dropout				| Drop 50%										|
| Convolution 5x5     	| 2x2 stride, same padding, outputs 12X40X48	|
| RELU					|												|
| Convolution 3x3	    | 2x2 stride, same padding, outputs 6x20x64  	|
| RELU					|												|
| Convolution 3x3	    | 2x2 stride, same padding, outputs 3x10x64  	|
| RELU					|												|
| Dropout				| Drop 50%										|
| Fully connected, 500  | outputs 100 									|
| RELU					|												|
| Fully connected, 100  | outputs 50 									|
| RELU					|												|
| Fully connected, 50   | 1 output - steering angle						|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two-four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the center of the lane. These images show what a recovery looks like starting from left to the right and from right back to center of the lane :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help model to learn better and generalize better. Here below I'll show what kind of images were used for training. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had approx. 6*5700 number of data points. I then preprocessed this data by cropping top 50 and bottem 20 pixels, which could disturb learning.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by trainig loss around 0.0124 and validation loss of 0.0126. More epochs makes loss slower but driving is worse in that case. I used an adam optimizer so that manually training the learning rate wasn't necessary.
