import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats

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
    m = test_samples.shape[0]
    n = training_data.shape[0]

    testSamples = test_samples.toarray()
    trainingData = training_data.toarray()

    testDots = (testSamples*testSamples).sum(axis=1).reshape((m,1))*np.ones(shape=(1,n))
    trainDots = (trainingData*trainingData).sum(axis=1)*np.ones(shape=(m,1))
    pairwise_distance = testDots + trainDots -2*testSamples.dot(trainingData.T)

    zero_mask = np.less(pairwise_distance, 0.0)
    pairwise_distance[zero_mask] = 0.0
    pairwise_distance = np.sqrt(pairwise_distance)
    
    # Find the k nearest neighbours of each samples as an m x k matrix of indices.
    nearest_neighbours = np.argpartition(pairwise_distance, k)[:, :k]
    
    # Look up the classes corresponding to each index.
    nearest_labels = training_labels[nearest_neighbours]
    
    # Return the most frequent class on each row.
    # Note: Ensure that the returned vector does not contain any empty dimensions.
    most_frequent_labels = scipy.stats.mode(nearest_labels, axis=1)[0]
    return np.squeeze(most_frequent_labels)

# Exp 1
for i in range(20):
    indexes = sample_indices(labels, 80, 80, 80, 80)
    trainingSample = data[indexes]
    testSample = scipy.sparse.csr_matrix(np.delete(data.todense(), indexes, 0))
    predictedOutputs = knn_classify(trainingSample, testSample, labels)
    actualOutputs = labels[indexes]
    for j in range(len(labels)):
        TP = np.sum((predictedOutputs == j) & (actualOutputs == j))
        TN = np.sum((predictedOutputs != j) & (actualOutputs != j))
        FP = np.sum((predictedOutputs == 1) & (actualOutputs == 0))
        FN = np.sum((predictedOutputs == 0) & (actualOutputs == 1))
    accuracy = truePositives / data.shape[0]
    correlation_coef = np.corrcoef(predictedOutputs.flatten(), actualOutputs.flatten())[0, 1]
    similarity_percentage = (correlation_coef + 1) / 2 * 100
    print("Test "+str(i)+" Accuracy of "+str(similarity_percentage))

# Exp 1 but shorter
# for i in range(20):
#     indexes = sample_indices(labels, 80, 80, 80, 80)
#     correlation_coef = np.corrcoef(knn_classify(data[indexes], scipy.sparse.csr_matrix(np.delete(data.todense(), indexes, 0)), labels).flatten(), labels[indexes].flatten())[0, 1]
#     print("Test "+str(i)+" Accuracy of "+str((correlation_coef + 1) / 2 * 100))