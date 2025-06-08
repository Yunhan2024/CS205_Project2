import numpy as np
import time
import json


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
    prev_accuracy = 0
    results = []

    print("Beginning search.")
    print()

    for i in range(n_features):
        feature_to_add = None
        best_so_far = 0

        for k in range(n_features):
            if k not in current_features:
                # Try adding feature k
                accuracy = leave_one_out_validation(data, current_features + [k])
                feature_list = ','.join(map(lambda x: str(x + 1), current_features + [k]))
                print(f"\tUsing feature(s) {{{feature_list}}} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_so_far:
                    best_so_far = accuracy
                    feature_to_add = k

        if feature_to_add is not None:
            current_features.append(feature_to_add)
            feature_list = ','.join(map(lambda x: str(x + 1), current_features))

            if best_so_far < prev_accuracy and i > 0:
                print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

            print(f"Feature set {{{feature_list}}} was best, accuracy is {best_so_far * 100:.1f}%")
            print()

            # Save result
            results.append({
                "features": f"{{{feature_list}}}",
                "accuracy": best_so_far * 100,
                "step": i + 1,
                "is_best": False
            })

            if best_so_far > best_accuracy:
                best_accuracy = best_so_far
                best_features = current_features.copy()

            prev_accuracy = best_so_far

    # Mark best result
    if results:
        best_idx = max(range(len(results)), key=lambda x: results[x]["accuracy"])
        results[best_idx]["is_best"] = True

    return best_features, best_accuracy, results


def backward_elimination(data):
    n_features = data.shape[1] - 1
    current_features = list(range(n_features))
    best_features = current_features.copy()
    best_accuracy = leave_one_out_validation(data, current_features)
    prev_accuracy = best_accuracy
    results = []

    print("Beginning search.")
    print()

    feature_list = ','.join(map(lambda x: str(x + 1), current_features))
    print(f"Using feature(s) {{{feature_list}}} accuracy is {best_accuracy * 100:.1f}%")
    print()

    results.append({
        "features": f"{{{feature_list}}}",
        "accuracy": best_accuracy * 100,
        "step": 0,
        "is_best": False
    })

    step = 1
    while len(current_features) > 1:
        feature_to_remove = None
        best_so_far = 0

        for k in current_features:
            features_minus_k = [f for f in current_features if f != k]
            accuracy = leave_one_out_validation(data, features_minus_k)
            feature_list = ','.join(map(lambda x: str(x + 1), features_minus_k))
            print(f"\tUsing feature(s) {{{feature_list}}} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_so_far:
                best_so_far = accuracy
                feature_to_remove = k

        if feature_to_remove is not None:
            current_features.remove(feature_to_remove)
            feature_list = ','.join(map(lambda x: str(x + 1), current_features))

            if best_so_far < prev_accuracy:
                print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

            print(f"Feature set {{{feature_list}}} was best, accuracy is {best_so_far * 100:.1f}%")
            print()

            results.append({
                "features": f"{{{feature_list}}}",
                "accuracy": best_so_far * 100,
                "step": step,
                "is_best": False
            })

            if best_so_far > best_accuracy:
                best_accuracy = best_so_far
                best_features = current_features.copy()

            prev_accuracy = best_so_far
            step += 1

    # Mark best result
    if results:
        best_idx = max(range(len(results)), key=lambda x: results[x]["accuracy"])
        results[best_idx]["is_best"] = True

    return best_features, best_accuracy, results


def main():
    try:
        filename = "CS205_small_Data__1.txt"
        classes, features_data = load_data(filename)
        if len(features_data.shape) == 1:
            features_data = features_data.reshape(-1, 1)
        data = np.column_stack((classes, features_data))

        # Choose algorithm
        print("Type the number of the algorithm you want to run.")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        choice = input()

        if choice == "1":
            best_features, best_accuracy, results = forward_selection(data)
            algorithm_name = "Forward Selection"
            filename_suffix = "forward_selection"
        elif choice == "2":
            best_features, best_accuracy, results = backward_elimination(data)
            algorithm_name = "Backward Elimination"
            filename_suffix = "backward_elimination"
        else:
            print("Invalid choice!")
            return

        feature_list = ','.join(map(lambda x: str(x + 1), best_features))
        print(
            f"Finished search!! The best feature subset is {{{feature_list}}}, which has an accuracy of {best_accuracy * 100:.1f}%")

        # Save results to JSON
        json_data = {
            "title": f"{algorithm_name} Results",
            "results": results,
            "best_result": {
                "features": f"{{{feature_list}}}",
                "accuracy": best_accuracy * 100
            }
        }

        output_filename = f'{filename_suffix}_results.json'
        with open(output_filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Results saved to {output_filename}")

    except:
        print("Error")


if __name__ == "__main__":
    main()