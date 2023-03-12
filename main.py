import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats
import threading

data, labels, class_names, vocabulary = np.load("ReutersNews_4Classes_sparse.npy", allow_pickle=True)

def sample_indices(labels, *num_per_class):
    """
    Returns randomly selected indices. It will return the specified number of indices for each class.
    """
    indices = []
    for cls, num in enumerate(num_per_class):
        cls_indices = np.where(labels == cls)[0]
        indices.extend(np.random.choice(cls_indices, size=num, replace=False))
    return np.array(indices)

def knn_classify(test_samples, training_data, training_labels, metric="euclidean", k=1):
    """
    Performs k-nearest neighbour classification on the provided samples,
    given training data and the corresponding labels.
    
    test_samples: An m x d matrix of m samples to classify, each with d features.
    training_data: An n x d matrix consisting of n training samples, each with d features.
    training_labels: A vector of size n, where training_labels[i] is the label of training_data[i].
    metric: The metric to use for calculating distances between samples.
    k: The number of nearest neighbours to use for classification.
    
    Returns: A vector of size m, where out[i] is the predicted class of test_samples[i].
    """
    # Calculate an m x n distance matrix.
    if(metric=="euclidean"):
        testSamples = test_samples.toarray()
        trainingData = training_data.toarray()
        testDots = np.square(testSamples).sum(axis=1).reshape((testSamples.shape[0],1))
        trainDots = np.square(trainingData).sum(axis=1)
        pairwise_distance = testDots + trainDots -2*testSamples.dot(trainingData.T)
        pairwise_distance = np.sqrt(pairwise_distance)
    elif(metric=="cosine"):
        dot_products = np.dot(test_samples, training_data.T)

        testSamples = test_samples.toarray()
        trainingData = training_data.toarray()

        test_norm = np.sqrt(np.sum(np.square(testSamples), axis=1)).reshape((testSamples.shape[0],1))
        training_norm = np.sqrt(np.sum(np.square(trainingData), axis=1))
        print("Shape of test: " +str(test_norm.shape)+" Shape of train: "+str(training_norm.shape))

        norm_products = test_norm * training_norm
        
        pairwise_similarity = dot_products / norm_products
        pairwise_distance = 1 - pairwise_similarity
    
    # Find the k nearest neighbours of each samples as an m x k matrix of indices.
    nearest_neighbours = np.argpartition(pairwise_distance, k)[:, :k]
    
    # Look up the classes corresponding to each index.
    nearest_labels = training_labels[nearest_neighbours]
    
    # Return the most frequent class on each row.
    # Note: Ensure that the returned vector does not contain any empty dimensions.
    most_frequent_labels = scipy.stats.mode(nearest_labels, axis=1, keepdims=True)[0]
    return np.squeeze(most_frequent_labels)

# Exp 1
def runTest(accuracies, metric="euclidean", k=1):
    trainIndexes = sample_indices(labels, 80, 80, 80, 80)
    testIndexes = np.delete(range(800),trainIndexes)
    trainingSample = data[trainIndexes]
    testSample = scipy.sparse.csr_matrix(np.delete(data.todense(), trainIndexes, 0))
    predictedOutputs = knn_classify(testSample, trainingSample, labels[trainIndexes], metric=metric, k=k)
    actualOutputs = labels[testIndexes]
    accuracy = np.sum(predictedOutputs == actualOutputs)/predictedOutputs.shape[0]
    accuracies.append(accuracy)

# accuracies = []
# for i in range(20):
#     runTest(accuracies, metric="euclidean", k=1)
# print("Test accuracy of "+str(np.mean(accuracies))+" and std deviation "+str(np.std(accuracies)))

# accuracies = []
# for i in range(20):
#     runTest(accuracies, metric="cosine", k=1)
#     print("Progress "+("#"*(i)), end="\r")
# print("Test accuracy of "+str(np.mean(accuracies))+" and std deviation "+str(np.std(accuracies)))

# Exp 2
for j in range(1,51):
    accuracies = []
    for i in range(20):
        runTest(accuracies, metric="cosine", k=j)
    # print("Test accuracy of "+str(np.mean(accuracies))+" and std deviation "+str(np.std(accuracies)))
    # print("Progress "+("#"*(j)), end="\r")

# Exp 3
def runTest2(accuracies, metric="euclidean", k=1):
    trainIndexes = sample_indices(labels, 100, 100, 100, 100)
    testIndexes = np.delete(range(800),trainIndexes)
    trainingSample = data[trainIndexes]
    testSample = scipy.sparse.csr_matrix(np.delete(data.todense(), trainIndexes, 0))
    predictedOutputs = knn_classify(testSample, trainingSample, labels[trainIndexes], metric=metric, k=k)
    actualOutputs = labels[testIndexes]
    accuracy = np.sum(predictedOutputs == actualOutputs)/predictedOutputs.shape[0]
    accuracies.append(accuracy)