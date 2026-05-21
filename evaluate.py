# evaluate.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from train import IoUMetric
from utils import combined_loss
from dataset import build_datasets

def evaluate(model, dataset, threshold=0.5):
    pixel_acc_list, iou_list = [], []

    for imgs, masks in dataset:
        preds = model.predict(imgs, verbose=0)
        preds_bin = (preds > threshold).astype(np.float32)
        masks_np  = masks.numpy()

        # Pixel accuracy
        correct = np.sum(preds_bin == masks_np)
        total   = masks_np.size
        pixel_acc_list.append(correct / total)

        # IoU
        intersection = np.sum(preds_bin * masks_np)
        union        = np.sum(preds_bin) + np.sum(masks_np) - intersection
        iou_list.append(intersection / (union + 1e-6))

    print(f"Pixel Accuracy : {np.mean(pixel_acc_list):.4f}")
    print(f"IoU            : {np.mean(iou_list):.4f}")
    return np.mean(pixel_acc_list), np.mean(iou_list)


def visualize_prediction(model, dataset, idx=0, threshold=0.5):
    """Affiche image / vérité terrain / prédiction pour le idx-ième exemple."""
    for i, (imgs, masks) in enumerate(dataset.unbatch().batch(1)):
        if i < idx:
            continue

        pred = model.predict(imgs, verbose=0)
        pred_bin = (pred[0, ..., 0] > threshold).astype(np.float32)

        img  = imgs.numpy()[0, ..., 0]
        mask = masks.numpy()[0, ..., 0]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img,      cmap="gray"); axes[0].set_title("Image")
        axes[1].imshow(mask,     cmap="gray"); axes[1].set_title("Vérité terrain")
        axes[2].imshow(pred_bin, cmap="gray"); axes[2].set_title("Prédiction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        break

if __name__ == "__main__":
    model = keras.models.load_model(
        "best_unet.keras",
        custom_objects={"combined_loss":combined_loss, "IoUMetric": IoUMetric}
    )
    _, val_ds = build_datasets("data/rimes_raw/DVD1_bis", "data/rimes_masks")

    evaluate(model, val_ds)
    visualize_prediction(model, val_ds, idx=0)