import numpy as np
import time


def load_data(filename):
    data = np.loadtxt(filename)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:]


def nearest_neighbor(train_data, test_instance, features):
    distances = np.sqrt(np.sum((train_data[:, features] - test_instance[features]) ** 2, axis=1))
    return np.argmin(distances)


def leave_one_out_validation(data, features):
    classes = data[:, 0]
    instances = data[:, 1:]
    correct = 0

    for i in range(len(data)):
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[i] = False
        nn_idx = nearest_neighbor(instances[train_mask], instances[i], features)
        if classes[train_mask][nn_idx] == classes[i]:
            correct += 1

    return correct / len(data)


def main():
    filename = "CS205_small_Data__1.txt"
    classes, features_data = load_data(filename)

    if len(features_data.shape) == 1:
        features_data = features_data.reshape(-1, 1)

    data = np.column_stack((classes, features_data))

    print(f"Data has {features_data.shape[1]} features, {len(classes)} instances.")

    # Test nearest_neighbor
    if data.shape[0] > 1:
        test_instance_idx = 0
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[test_instance_idx] = False
        all_feature_indices = np.arange(features_data.shape[1])
        nn_idx = nearest_neighbor(features_data[train_mask], features_data[test_instance_idx], all_feature_indices)
        print(f"NN for 1:  {np.where(train_mask)[0][nn_idx] + 1}.")

    # Test leave_one_out_validation
    features_to_test = [0, 1]
    if features_data.shape[1] < len(features_to_test):
        features_to_test = np.arange(features_data.shape[1]).tolist()

    if len(features_to_test) > 0:
        accuracy = leave_one_out_validation(data, features_to_test)
        print(f"LOO accuracy (features {features_to_test}): {accuracy:.4f}")


if __name__ == "__main__":
    main()
