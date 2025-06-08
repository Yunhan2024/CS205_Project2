import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'


def visualize_forward_selection():
    with open('forward_selection_results.json', 'r') as f:
        data = json.load(f)

    results = data['results']
    first_three = results[:3]
    last_three = results[-3:]

    best_result = next((result for result in results if result['is_best']), None)

    features = []
    accuracies = []
    colors = []

    for item in first_three:
        features.append(item['features'])
        accuracies.append(item['accuracy'])
        colors.append('lightblue' if item['is_best'] else 'gray')

    middle_count = 3
    start_acc = first_three[-1]['accuracy']
    end_acc = last_three[0]['accuracy']
    middle_accs = np.linspace(start_acc, end_acc, middle_count + 2)[1:-1]

    for i, acc in enumerate(middle_accs):
        if i == 1:  # Middle bar gets the omitted text
            features.append('{omitted for space}')
        else:
            features.append('...')
        accuracies.append(acc)
        colors.append('lightgray')

    for item in last_three:
        features.append('')
        accuracies.append(item['accuracy'])
        colors.append('lightblue' if item['is_best'] else 'gray')

    plt.figure(figsize=(12, 6))

    bars = plt.bar(range(len(features)), accuracies, color=colors, width=0.4)

    plt.axvline(x=2.5, color='black', linestyle=':', linewidth=3, alpha=1.0)
    plt.axvline(x=5.5, color='black', linestyle=':', linewidth=3, alpha=1.0)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold',
                 fontfamily='Times New Roman', fontsize=14)

    plt.xlabel('Feature Set', fontfamily='Times New Roman', fontsize=16)
    plt.ylabel('Accuracy (%)', fontfamily='Times New Roman', fontsize=16)
    plt.title('Forward Selection Results - Small dataset', fontfamily='Times New Roman', fontsize=18)
    plt.xticks(range(len(features)), features, rotation=0, ha='center', fontfamily='Times New Roman', fontsize=12)
    plt.yticks(fontfamily='Times New Roman', fontsize=12)

    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Best Result: {best_result['features']} with {best_result['accuracy']:.1f}% accuracy")
    print(f"Showing first 3, last 3, and best result from {len(results)} total steps")


visualize_forward_selection()