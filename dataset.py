# dataset.py
import numpy as np
from PIL import Image
import albumentations as A
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

PATCH_SIZE = 512

train_aug = A.Compose([
    A.RandomCrop(PATCH_SIZE, PATCH_SIZE),
    A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=34, sigma=4, p=0.5),
])

val_aug = A.Compose([
    A.CenterCrop(PATCH_SIZE, PATCH_SIZE),
])


def load_and_augment(img_path, mask_path, training):
    """Charge image + masque, augmente, normalise. Retourne (H,W,1) float32."""
    img_path  = img_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    img  = np.array(Image.open(img_path).convert("L"))   # (H, W)
    mask = np.array(Image.open(mask_path).convert("L"))  # (H, W)

    aug = train_aug if training else val_aug
    out = aug(image=img, mask=mask)
    img, mask = out["image"], out["mask"]

    # Normalisation : µ=0.5, σ=0.25
    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.25

    # Masque binaire [0,1]
    mask = (mask > 127).astype(np.float32)

    # Channels-last (H, W, 1)
    return img[..., np.newaxis], mask[..., np.newaxis]


def make_tf_dataset(img_paths, mask_paths, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if training:
        ds = ds.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)

    AUTOTUNE = tf.data.AUTOTUNE

    def _map_fn(img_p, mask_p):
        img, mask = tf.py_function(
            func=lambda i, m: load_and_augment(i, m, training),
            inp=[img_p, mask_p],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape([PATCH_SIZE, PATCH_SIZE, 1])
        mask.set_shape([PATCH_SIZE, PATCH_SIZE, 1])
        return img, mask

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_datasets(data_dir, mask_dir, batch_size=8):
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    img_paths  = [os.path.join(data_dir, os.path.splitext(f)[0] + ".tif")
                  for f in mask_files]

    # Filtre : garde seulement les paires où l'image existe
    pairs = [(i, m) for i, m in zip(img_paths, mask_paths) if os.path.exists(i)]
    img_paths, mask_paths = zip(*pairs)

    img_tr, img_val, mask_tr, mask_val = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_ds = make_tf_dataset(img_tr, mask_tr, batch_size, training=True)
    val_ds   = make_tf_dataset(img_val, mask_val, batch_size, training=False)
    return train_ds, val_ds