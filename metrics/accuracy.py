import numpy as np


def one_off_accuracy(y_true, y_pred):
    true_indices = np.argmax(y_true, axis=-1)
    hits = 0
    for i, prediction in enumerate(y_pred):
        max_index = len(prediction) - 1
        left_neighbor = 0 if true_indices[i] == 0 else true_indices[i] - 1
        right_neighbor = max_index if true_indices[i] == max_index else true_indices[i] + 1
        hits += 1 if 1 in np.array(prediction)[np.array([true_indices[i], left_neighbor, right_neighbor])] else 0
    return hits / len(y_true)
