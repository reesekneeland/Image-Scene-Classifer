# Image-Scene-Classifer
An algorithm that uses SVM and BoWs techniques to classify an image as one of a selection of scene classificationis.

### Tiny Image KNN Classification
![Figure_1](https://user-images.githubusercontent.com/77468346/172687642-5e5f313a-55b7-4ce7-80c7-6b82bdfa6538.png)


The **classify_knn_tiny()** function is the driver function for the scene classification algorithm using tiny image feature selection and KNN predictions. For every sample, it feeds the images into the **get_tiny_image()** function, which resizes the image to a tiny representation with size (16, 16), subtracts the mean, and divides by the norm to normalize the tiny image into a feature we can classify. Our driver function then calls **predict_knn()** which applies the sklearn NearestNeighbors KNN classifier to generate a vector of predicted labels for the given features. With a KNN algorithm using k=8, a tiny image size of (16x16), I was able to classify scene images with 22.1% accuracy.

### BoW KNN Classification
![Figure_2](https://user-images.githubusercontent.com/77468346/172687650-f7285a3f-0f5b-4787-bd0e-3942c66ebd9d.png)


The **classify_bow_knn()** function is the driver function for our scene classification algorithm using Bag of Words feature selection and KNN predictions, it generates a list of vocab words from a scene using the **build_visual_dictionary()** function (loading in the vocab words if already generated, as this is a long process). The driver function feeds the data into the **compute_dsift()** function, which extracts the SIFT features, then feeds those features and the list of vocab words into the **compute_bow()** function, which again uses the sklearn NearestNeighbor function to find the nearest cluster center for each sample. The cluster samples are then returned to the driver function which finally calls **predict_knn()** again to generate the vector of predictions. Using dic_size=300, k=16, stride=25, and size=25, I was able to classify scene images with 54.0% accuracy.

### BoW SVM Classification
![Figure_3](https://user-images.githubusercontent.com/77468346/172687661-e6ebbdf7-08b4-4dc5-b5fe-020b365168a3.png)


The **classify_bow_svm()** function is the driver function for our scene classification algorithm using Bag of Words feature selection and SVM predictions. It behaves exactly like the previous algorithm, using the same vocab list, up until the prediction step, in which it feeds the generated BoW features into the **predict_svm()** function. This generates 15 1-vs-all implementations of the sklearn LinearSVC function, giving me binary probabilities for each class in respect to each sample. It then takes all of those predictions and selects the class with the highest binary prediction odds as the output class, returning a vector of predicted classes for each sample. Using the same parameters as **classify_bow_knn()** for the first steps, C=1 in the SVM algorithm, and n_classes=15 for the SVM predictions, I was able to classify scene images with 62.9% accuracy.
