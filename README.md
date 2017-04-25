# Traffic Sign Recognition Project


[//]: # (Image References)

[image1]: ./images/signs_samples.png "Signs sample"
[image2]: ./images/signs_distribution.png "Signs distribution"
[image3]: ./images/transform_samples.png "Transformed images"
[image4]: ./images/samples.png "Samples"
[image5]: ./images/processed_signs.png "Processed signs"
[image6]: ./web_samples/sample1.png "Traffic Sign 1"
[image7]: ./web_samples/sample2.png "Traffic Sign 2"
[image8]: ./web_samples/sample3.png "Traffic Sign 3"
[image9]: ./web_samples/sample4.png "Traffic Sign 4"
[image10]: ./web_samples/sample5.png "Traffic Sign 5"
[image11]: ./web_samples/sample6.png "Traffic Sign 6"
[image12]: ./images/predictions.png "Predictions"
[image13]: ./images/softmax_predictions.png "Softmax predictions"


### Data Set Summary & Exploration

Numpy library was used to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset.

Here is an exploratory visualization of the data set.  We can explore samples of avaliable classes below:  

![alt text][image1]

As we can see data samples are not uniformly distributed across all classes. Some classes have a lot more data samples. Also distributions are close for training, validation and test sets (sligtly different for validation set). Becouse of this fact non-uniform distribution should not be a problem for the existing data. However, there is a risk that model will not be generalized well for additional data (for example samples from the web) as it can be overtrained for specific classes. We will try to address this issue during preprocessing step later.

![alt text][image2]

### Pre-process image data 

1. As a first step, I decided to deal with non-unifromly distributed data. The idea was to make balanced training set where all classes are represented with the same number of samples. I decided to make copies from the existing images with low number of samples. Additionaly, to prevent exact same images appearing twice two addtional transformations were selected to add some variation to the resulting set:
* Randomly scaling image up with factor 1.0 - 1.3
* Randomly rotating image wit angle from -15 to 15 degree.

 These parameters above just worked best for me with some other choices. 
 As the result total number of trainig samples increased to 86430
 
 Here is a sample of fake images generated using transforms provided:
 
![alt text][image3]

2. Also to extend ability for the generalization I decided to extend data set using same transformation above for whole dataset two more times. As the result total number of training sample increased to 259290 (6030 images per class)

3. Analyzing images futher I found that there is a huge variety of different contrast levels for the images:

![alt text][image4]

 Trying to deal with this issue and increase validation accuracy I used different techiques like converting images to different color    spaces, histogram equalization and ect. One which appeared to work best is CLAHE (Contrast Limited Adaptive Histogram Equalization)(http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)  
 
 Here is how processed images looks like:

![alt text][image5]

4. The last step is a to normalize images. Intersting fact to mention there is that pixel / 255  normalization works significaly better than (pixel - 128) / 128 proposed as sample.  

### Model Architecture  

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   							| 
| Convolution 4x4     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 |
| Convolution 4x4     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 |
| Convolution 4x4     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 |
| Fully connected		| 2048x512        									|
| Dropout | keep_prob = 0.5        									|
| Fully connected		| 512x128        									|
| Dropout | keep_prob = 0.5        									|
| Fully connected		| 128x43        									|
| Softmax				|       									|

1. After pre-processing steps are done I tried initial LeNet model as base - and results were sligtly above 90% accuracy for validation set. It appeared that adding  dropout layer for the first fully connected layer with keep_probability = 0.5 could do the trick. Just with this small change I was able to archive 94% accuracy for the validation set with base LeNet model. 

2. To archive higher accuracy results, I tried following options:
 * Adding new features maps to convolutional layers
 * Playng with padding types and filter sizes, tried following filter sizes: 4, 5, 6. Results are pretty close there. selected 4x4 filters for the model
 * Playing with fully connected layers sizes.
 * Finaly decided to add one more fully connected layer - it slightly increased perfoemance for the model.

3. For the training part of the model I decided to select AdamOptimizer as it uses momentum and adaptive learning rates and should work better when SDC.
  * While playing with batch sizes I relized that 128 batch size works slightly better than lager sizes as 256, 512, 1024. 
  * Trying different learning rate values (0.01, 0.001, 0.0005, 0.0001). I realized that 0.0005 works best for my model.
  * Also it looks like 50 - 100 epochs is enough for model to stop increasing validation accuracy reaching traing accuracy ~99.9%
 
After model is selected and tuned my final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.1%
* test set accuracy of 96.0%

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

I expected that two images might be difficult to classify:
 * Children crossing - becouse of shadow and noisy background.
 * Beware Ice/Snow - becose of snow noise

Here is predictions results:

[alt text][image12]

Results are a bit surprising. The predicion is wrong for "bumpy road" sign which is a best quality sign :) 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

