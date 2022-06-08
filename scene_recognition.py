import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC, SVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list

def compute_dsift(img, stride, size):
    SIFT = cv2.xfeatures2d.SIFT_create()
    #extract features from SIFT key points
    k, dense_feature = SIFT.compute(img, [cv2.KeyPoint(x=i, y=j, size=size) for i in range(0, img.shape[1], stride) for j in range(0, img.shape[0], stride)])
    return dense_feature


def get_tiny_image(img, output_size):
    #resize to given output size
    img_small = cv2.resize(img, output_size)
    #compute normalized length and divide to get unit length
    img_small_centered = img_small-np.mean(img_small)
    norm = np.linalg.norm(img_small_centered)
    feature = img_small_centered/norm
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # initialize KNN algorithm
    KNN = NearestNeighbors(n_neighbors=k)
    KNN.fit(feature_train)
    dist, indMap = KNN.kneighbors(feature_test, n_neighbors=k)
    #check sizes
    if(len(dist) == len(indMap)):
        label_test_pred = []
        for i in range(len(indMap)):
            #generate predictions
            indSlice = indMap[i,:]
            curLabels = label_train[indSlice]
            labelCount = np.bincount(curLabels)
            prediction = np.argmax(labelCount)
            label_test_pred.append(prediction)
    else:
        print("array slices of different size")
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    #initialize values
    feature_train = np.zeros((len(img_train_list), 256))
    feature_test = np.zeros((len(img_train_list), 256))
    indexed_label_train = np.zeros((len(label_train_list), )).astype(int)
    indexed_label_test = np.zeros((len(label_train_list), )).astype(int)
    tiny_size = (16, 16)
    k = 8
    #construct feature train and test sets as well as label arrays for predict_knn
    for i, file in enumerate(img_train_list):
        feature_train[i] = get_tiny_image(cv2.imread(file, cv2.IMREAD_GRAYSCALE), tiny_size).reshape(-1)

    for i, file in enumerate(img_test_list):
        feature_test[i] = get_tiny_image(cv2.imread(file, cv2.IMREAD_GRAYSCALE), tiny_size).reshape(-1)

    for i, label in enumerate(label_train_list):
        indexed_label_train[i] = label_classes.index(label)

    for i, label in enumerate(label_test_list):
        indexed_label_test[i] = label_classes.index(label)
    predictions = predict_knn(feature_train, indexed_label_train, feature_test, k)

    #calculate confusion
    confusion = confusion_matrix(indexed_label_test, predictions)
    #calculate accuracy
    accuracy = accuracy_score(indexed_label_test, predictions)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    return KMeans(n_clusters=dic_size, n_init=10, max_iter=500).fit(dense_feature_list).cluster_centers_

def compute_bow(feature, vocab):
    # To do
    bow_count = [0] *  (vocab.shape[0])
    KNN = NearestNeighbors(algorithm='ball_tree')
    KNN.fit(vocab)
    distances, indMap = KNN.kneighbors(feature, n_neighbors=1)
    for i, c in enumerate(indMap):
        bow_count[int(c)] += 1
    norm = np.linalg.norm(bow_count)
    bow_feature = np.array(bow_count/norm)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    stride = 25
    size = 25
    dic_size = 300
    k = 16
    #initialize containers
    feature_list = []
    vocab_bow_train = np.zeros((len(label_train_list), dic_size))
    vocab_bow_test = np.zeros((len(label_train_list), dic_size))
    indexed_label_train = np.zeros((len(label_train_list), )).astype(int)
    indexed_label_test = np.zeros((len(label_train_list), )).astype(int)
    try:
        vocab = np.loadtxt("vocab.txt")
        print("loading vocab.txt")
    except:
        print("couldnt find vocab.txt")
        #generate initial dense features
        for i, file in enumerate(img_train_list):
            print("getting features for vocab" + str(i))
            feature_list.extend(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size))
        #generate vocab from dense features
        vocab = build_visual_dictionary(feature_list, dic_size)
        np.savetxt('vocab.txt', vocab)


    #generate train vocab bow list
    for i, file in enumerate(img_train_list):
        print("train: " + str(i))
        vocab_bow_train[i] = compute_bow(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size), vocab).flatten()

    #generate test vocab bow list from test img features
    for i, file in enumerate(img_test_list):
        print("test: " + str(i))
        vocab_bow_test[i] = compute_bow(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size), vocab).flatten()

    #generate label index lists
    for i, label in enumerate(label_train_list):
        indexed_label_train[i] = label_classes.index(label)

    for i, label in enumerate(label_test_list):
        indexed_label_test[i] = label_classes.index(label)
    #generate predictions
    predictions = predict_knn(vocab_bow_train, indexed_label_train, vocab_bow_test, k)

    #calculate confusion
    confusion = confusion_matrix(indexed_label_test, predictions)
    
    #calculate accuracy
    accuracy = accuracy_score(indexed_label_test, predictions)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    C = 1
    one_vs_all_models, model_predictions, label_test_pred = [], [], []
    #initialize 15 1-vs-all classifiers
    for i in range(n_classes):
        one_vs_all_models.append(LinearSVC(tol=1e-5, C=C))
    for i in range(n_classes):
        #generate a list of labels for this class
        model_labels = []
        for l in label_train:
            if l == i: 
                model_labels.append(1)
            else:
                model_labels.append(0)
        #fit each respective model to its labels
        one_vs_all_models[i].fit(feature_train, model_labels)
    for i, n in enumerate(one_vs_all_models):
        #generate predictions
        model_predictions.append(n.decision_function(feature_test))
    model_predictions = np.array(model_predictions)
    for i, n in enumerate(feature_test):
        #assemble the final prediction vector with the best predictions of each class
        label_test_pred.append(np.argmax(model_predictions[:,i,]))

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    stride = 25
    size = 25
    #c====3
    dic_size = 300
    n_classes = 15
    #initialize containers
    feature_list = []
    vocab_bow_train = np.zeros((len(label_train_list), dic_size))
    vocab_bow_test = np.zeros((len(label_train_list), dic_size))
    indexed_label_train = np.zeros((len(label_train_list), )).astype(int)
    indexed_label_test = np.zeros((len(label_train_list), )).astype(int)
    #load in the vocab list or generate one if not available
    try:
        vocab = np.loadtxt("vocab.txt")
        print("loading vocab.txt")
    except:
        print("couldnt find vocab.txt")
        #generate initial dense features
        for i, file in enumerate(img_train_list):
            print(i)
            feature_list.extend(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size))
        #generate vocab from dense features
        vocab = build_visual_dictionary(feature_list, dic_size)
        np.savetxt('vocab.txt', vocab)


    #generate train vocab bow list
    for i, file in enumerate(img_train_list):
        print("train: " + str(i))
        vocab_bow_train[i] = compute_bow(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size), vocab).flatten()

    #generate test vocab bow list from test img features
    for i, file in enumerate(img_test_list):
        print("test: " + str(i))
        vocab_bow_test[i] = compute_bow(compute_dsift(cv2.imread(file, cv2.IMREAD_GRAYSCALE), stride, size), vocab).flatten()

    #generate label index lists
    for i, label in enumerate(label_train_list):
        indexed_label_train[i] = label_classes.index(label)

    for i, label in enumerate(label_test_list):
        indexed_label_test[i] = label_classes.index(label)
    #generate predictions
    predictions = predict_svm(vocab_bow_train, indexed_label_train, vocab_bow_test, n_classes)

    #calculate confusion
    confusion = confusion_matrix(indexed_label_test, predictions)
    
    #calculate accuracy
    accuracy = accuracy_score(indexed_label_test, predictions)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)



