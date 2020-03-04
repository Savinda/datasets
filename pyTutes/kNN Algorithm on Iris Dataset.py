#!/usr/bin/env python
# coding: utf-8

# In[24]:


# KNN on Iris dataset
from csv import reader
from random import seed
from random import randrange
from math import sqrt


# In[25]:


# Load csv file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[26]:


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
# Convert string column into interger
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# In[27]:


# Find the min and max values for the each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# In[28]:


# Rescale dataset columns to the 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / minmax[i][1] - minmax[i][0]


# In[29]:


# Split dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# In[30]:


# Calculate accureacy percentage
def accuracy_metirc(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        return correct / float(len(actual)) * 100.0


# In[31]:


# Evaluate an algorithm using cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metirc(actual, predicted)
        scores.append(accuracy)
    return scores


# In[32]:


# Calculate the Euclidean distance between two vectors
def euclidean_distace(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i], row2[i]) ** 2
    return sqrt(distance)


# In[33]:


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distance = list()
    for train_row in train:
        dist = euclidean_distace(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key = lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# In[34]:


# Make prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# In[35]:


# kNN algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)


# In[36]:


# Test the kNN on the Iris dataset
seed(1)
filename = 'datasets/iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)


# In[37]:


# Convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)


# In[38]:


# Evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)


# In[39]:


print('Mean Accuracy: %.3f%%' % (sum(scores)/ float(len(scores))))

