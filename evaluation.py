"""
evaluation.py

Provides functions for evaluating model performance, including confusion matrix visualization for each output label in the classification chain.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrices(y_true_df, predictions, label_encoders, title_suffix=""):
    """
    Plot confusion matrices for each output label in a chained multi-output classification.

    :param y_true_df: DataFrame of true labels (encoded) with columns matching label_encoders keys.
    :param predictions: numpy array of predicted labels (encoded) with shape (n_samples, n_labels).
    :param label_encoders: Dictionary of LabelEncoders for each label column (to get class names).
    :param title_suffix: Optional suffix to include in plot titles (e.g., embedding type).
    """
    labels = list(label_encoders.keys())
    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4))
    if n_labels == 1:
        axes = [axes]  # ensure axes is iterable even if only one label

    for i, label in enumerate(labels):
        true_vals = y_true_df[label]
        pred_vals = predictions[:, i]
        # Retrieve class names from the encoder
        classes = label_encoders[label].classes_
        # Compute confusion matrix for the label
        cm = confusion_matrix(true_vals, pred_vals, labels=range(len(classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=axes[i], xticks_rotation='vertical', colorbar=False)
        axes[i].set_title(f"Confusion Matrix for {label.upper()}" + (f" ({title_suffix})" if title_suffix else ""))

    # If a title_suffix is provided, add a suptitle and adjust layout accordingly.
    if title_suffix:
        fig.suptitle(f"Embedding: {title_suffix}", fontsize=14)
        # Reserve space at the top for the suptitle (adjust the rect as needed)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        plt.tight_layout()

    # Optionally adjust the bottom margin if labels are clipped
    plt.subplots_adjust(bottom=0.15)
    plt.show()

