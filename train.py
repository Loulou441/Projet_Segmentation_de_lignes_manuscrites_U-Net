# train.py
import tensorflow as tf
from tensorflow import keras
from utils import combined_loss, IoUMetric
from model import build_unet
from dataset import build_datasets

class IoUMetric(keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold    = threshold
        self.intersection = self.add_weight(name="intersection", shape=(), initializer="zeros")
        self.union        = self.add_weight(name="union",        shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.union.assign_add(
            tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - tf.reduce_sum(y_true * y_pred)
        )

    def result(self):
        return self.intersection / (self.union + 1e-6)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


def train(data_dir="data/rimes_raw/DVD1",
          mask_dir="data/rimes_masks",
          batch_size=8,
          epochs=30,
          base_features=64):

    train_ds, val_ds = build_datasets(data_dir, mask_dir, batch_size)

    model = build_unet(base_features=base_features)

    steps = len(list(train_ds))  # nombre de batches par époque
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=epochs * steps
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[IoUMetric()]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_unet.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir="logs/"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history


if __name__ == "__main__":
    model, history = train()