import numpy as np
import time


def load_data(filename):
    data = np.loadtxt(filename)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:]  # classes, features


def nearest_neighbor(train_data, test_instance, features):
    distances = np.sqrt(np.sum((train_data[:, features] - test_instance[features]) ** 2, axis=1))
    return np.argmin(distances)


def leave_one_out_validation(data, features):
    classes, instances = data[:, 0], data[:, 1:]
    correct = 0

    for i in range(len(data)):
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[i] = False

        nn_idx = nearest_neighbor(instances[train_mask], instances[i], features)
        if classes[train_mask][nn_idx] == classes[i]:
            correct += 1

    return correct / len(data)



def main():
    print("Welcome to Feature Selection Algorithm")
    filename = input("Type the name of the file to test: ")

    try:
        # Load data
        classes, features = load_data(filename)

        # Debug output
        print(f"\nData loaded successfully!")
        print(f"Classes shape: {classes.shape}")
        print(f"Features shape: {features.shape}")

        # Check if features is 1D (only one feature)
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        data = np.column_stack((classes, features))

        print(f"\nThis dataset has {features.shape[1]} features with {len(classes)} instances.\n")

        # Normalize features
        for i in range(1, data.shape[1]):
            data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])




if __name__ == "__main__":
    main()