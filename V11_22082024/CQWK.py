import numpy as np
from sklearn.metrics import cohen_kappa_score

confusion_matrix = np.array([
    [727, 0, 0, 0],
    [0, 442, 2, 0],
    [0, 0, 1881, 0],
    [5, 0, 7, 115]
])

# Custom Weight Matrix (8, 4, 2, 1)
custom_weights = [8, 4, 2, 1]
n_classes = len(custom_weights)
weight_matrix = np.zeros_like(confusion_matrix, dtype=float)

for i in range(n_classes):
    for j in range(n_classes):
        weight_matrix[i, j] = custom_weights[i]

# Gewichtete Konfusionsmatrix
weighted_confusion_matrix = confusion_matrix * weight_matrix

# Rekonstruieren der tatsächlichen und vorhergesagten Labels basierend auf der gewichteten Matrix
true_labels_weighted = []
pred_labels_weighted = []

labels = ["RED", "AMBER", "GREEN", "WHITE"]

for i in range(len(labels)):
    for j in range(len(labels)):
        true_labels_weighted.extend([labels[i]] * int(weighted_confusion_matrix[i, j]))
        pred_labels_weighted.extend([labels[j]] * int(weighted_confusion_matrix[i, j]))

# QWK berechnen
qwk_weighted = cohen_kappa_score(true_labels_weighted, pred_labels_weighted, weights='quadratic')

# Ergebnis ausgeben
print("Quadratic Weighted Kappa (QWK) nach Anwendung der Gewichtung:", qwk_weighted)

