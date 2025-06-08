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
    """Evaluate accuracy using leave-one-out cross-validation"""
    classes, instances = data[:, 0], data[:, 1:]
    correct = 0

    for i in range(len(data)):
        # Leave one out
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[i] = False

        # Find nearest neighbor
        nn_idx = nearest_neighbor(instances[train_mask], instances[i], features)
        if classes[train_mask][nn_idx] == classes[i]:
            correct += 1

    return correct / len(data)


def forward_selection(data):

    n_features = data.shape[1] - 1
    current_features = []
    best_features = []
    best_accuracy = 0

    print("Forward Selection Begins:")

    for i in range(n_features):
        feature_to_add = None
        best_so_far = 0

        for k in range(n_features):
            if k not in current_features:
                # Try adding feature k
                accuracy = leave_one_out_validation(data, current_features + [k])
                print(
                    f"\tUsing features {{{', '.join(map(lambda x: str(x + 1), current_features + [k]))}}} accuracy is {accuracy:.4f}")

                if accuracy > best_so_far:
                    best_so_far = accuracy
                    feature_to_add = k

        if feature_to_add is not None:
            current_features.append(feature_to_add)
            print(
                f"\nFeature set {{{', '.join(map(lambda x: str(x + 1), current_features))}}} was best.\n")

            if best_so_far > best_accuracy:
                best_accuracy = best_so_far
                best_features = current_features.copy()

    return best_features, best_accuracy


def main():
    try:
        filename = "CS205_small_Data__1.txt"
        classes, features_data = load_data(filename)
        if len(features_data.shape) == 1:
            features_data = features_data.reshape(-1, 1)
        data = np.column_stack((classes, features_data))
        best_features, best_accuracy = forward_selection(data)
        print(f"Features: {best_features}, Accuracy: {best_accuracy:.4f}")
    except:
        print("Error")

if __name__ == "__main__":
    main()