import numpy as np
from sklearn.model_selection import train_test_split


def generate_adjacency_matrix(class_type):
    # Create a 10x10 matrix filled with zeros
    matrix = np.zeros((10, 10), dtype=int)

    if class_type == 0:
        # Class 0: More connections in the upper left quadrant
        matrix[:5, :5] = np.random.choice([0, 1], size=(5, 5), p=[0.3, 0.7])
    else:
        # Class 1: More connections in the lower right quadrant
        matrix[5:, 5:] = np.random.choice([0, 1], size=(5, 5), p=[0.3, 0.7])

    # Fill the rest of the matrix with random connections
    mask = matrix == 0
    matrix[mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.8, 0.2])

    # Ensure the matrix is symmetric (undirected graph)
    matrix = np.maximum(matrix, matrix.T)

    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(matrix, 0)

    return matrix


def generate_dataset(num_samples):
    X = []
    y = []

    for _ in range(num_samples):
        class_type = np.random.choice([0, 1])
        matrix = generate_adjacency_matrix(class_type)
        X.append(matrix)
        y.append(class_type)

    return np.array(X), np.array(y)


# Generate a dataset with 1000 samples
X, y = generate_dataset(1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Save the data to files
np.save('src/datasets/example_dataset_1/train_ds.npy', X_train)
np.save('src/datasets/example_dataset_1/test_ds.npy', X_test)
np.save('src/datasets/example_dataset_1/train_labels.npy', y_train)
np.save('src/datasets/example_dataset_1/test_labels.npy', y_test)

# Print shapes of the saved data
print(f"Train matrices shape: {X_train.shape}")
print(f"Test matrices shape: {X_test.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Print an example from each class
print("\nExample of Class 0:")
print(X_train[y_train == 0][0])
print("\nExample of Class 1:")
print(X_train[y_train == 1][0])

print("\nData has been saved to the following files:")
print("train_matrices.npy")
print("test_matrices.npy")
print("train_labels.npy")
print("test_labels.npy")
