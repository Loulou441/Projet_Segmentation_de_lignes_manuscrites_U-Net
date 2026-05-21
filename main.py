#Main code of the training process
import os
import tensorflow as tf
from tensorflow import keras
from utils import batch_generate_masks_rimes, combined_loss
from evaluate import evaluate, visualize_prediction
from dataset import build_datasets
from train import train, IoUMetric
from model import build_unet

if __name__ == "__main__":
    batch_generate_masks_rimes("data/rimes_raw/DVD1", "data/rimes_masks")
    model = build_unet()
    model.summary()
    model, history = train()
    model = keras.models.load_model(
        "best_unet.keras",
        custom_objects={"combined_loss":combined_loss, "IoUMetric": IoUMetric}
    )
    _, val_ds = build_datasets("data/rimes_raw/DVD1", "data/rimes_masks")

    evaluate(model, val_ds)
    visualize_prediction(model, val_ds, idx=0)