import math
from collections import Counter

data=[
    (1,3,'+'),
    (1,5, '+'),
    (2,3, 'o'),
    (2,4, '+'),
    (3,2, 'o'),
    (4,1, 'o')
]

def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def knn_predict(new_sample, data, k):
    distances = []
    for sample in data:
        dist = calculate_euclidean_distance(new_sample, sample)
        distances.append((dist, sample[2]))
    
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])
    
    # Get the labels of the k nearest neighbors
    k_nearest_labels = [label for _, label in distances[:k]]

    # Perform majority vote
    label_count = Counter(k_nearest_labels)
    return label_count.most_common(1)[0][0]



def cross_validation(data, k):
    errors = []

    for i in range(len(data)):
        test_sample = data[i]
        training_data = data[:i] + data[i+1:]

        predicted_label = knn_predict(test_sample, training_data, k)

        actual_label = test_sample[2]
        error = 1 if predicted_label != actual_label else 0
        errors.append(error)
        
        # Print the result for this fold
        print(f"Fold {i+1}: Test Sample {test_sample}, Predicted: {predicted_label}, Actual: {actual_label}, Error: {error}")

    loo_error = sum(errors)/len(errors)
    return loo_error

new_sample = (4,4)

# Predict using 1-NN
prediction_1nn = knn_predict(new_sample, data, k=1)
print(f"1-NN prediction: {prediction_1nn}")

# Predict using 3-NN
prediction_3nn = knn_predict(new_sample, data, k=3)
print(f"3-NN prediction: {prediction_3nn}")

# Perform LOO cross-validation for 1-NN without changing knn_predict
print("1-NN Leave-One-Out Cross-Validation:")
loo_error_1nn = cross_validation(data, k=1)
print(f"1-NN LOO Cross-Validation Error: {loo_error_1nn}\n")

# Perform LOO cross-validation for 3-NN without changing knn_predict
print("3-NN Leave-One-Out Cross-Validation:")
loo_error_3nn = cross_validation(data, k=3)
print(f"3-NN LOO Cross-Validation Error: {loo_error_3nn}")
    





