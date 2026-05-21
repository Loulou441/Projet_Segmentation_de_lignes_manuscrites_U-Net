# utils.py
import os
import numpy as np
from PIL import Image
from lxml import etree
import tensorflow as tf


def xml_to_mask_rimes(xml_path, img_path, out_path):
    """Parse le XML RIMES et génère un masque binaire PNG."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    img = Image.open(img_path).convert("L")
    W, H = img.size  # PIL retourne (width, height)
    mask = np.zeros((H, W), dtype=np.uint8)

    for box in root.iter("box"):
        # Coordonnées du bloc
        x0 = int(box.get("top_left_x", 0))
        y0 = int(box.get("top_left_y", 0))
        x1 = int(box.get("bottom_right_x", W))
        y1 = int(box.get("bottom_right_y", H))

        text_el = box.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text

        # Filtre : blocs trop petits (peu fiables)
        block_h = y1 - y0
        if block_h < 20:
            continue

        nb_lines = text.count("\n") + 1
        line_h = block_h / nb_lines
        dilation = 0.10  # 10 % de dilatation verticale

        for i in range(nb_lines):
            ly0 = int(y0 + i * line_h)
            ly1 = int(y0 + (i + 1) * line_h)
            pad = int((ly1 - ly0) * dilation)
            ly0 = max(0, ly0 - pad)
            ly1 = min(H, ly1 + pad)
            mask[ly0:ly1, x0:x1] = 255

    Image.fromarray(mask).save(out_path)


def batch_generate_masks_rimes(data_dir, mask_dir):
    """Génère tous les masques pour un dossier DVD."""
    os.makedirs(mask_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(data_dir) if f.endswith(".xml")]

    for xml_file in xml_files:
        stem = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(data_dir, xml_file)
        img_path = os.path.join(data_dir, stem + ".tif")
        out_path = os.path.join(mask_dir, stem + ".png")

        if not os.path.exists(img_path):
            print(f"Image manquante : {img_path}")
            continue

        xml_to_mask_rimes(xml_path, img_path, out_path)
        print(f"Masque généré : {out_path}")


# --- Dice Loss et perte combinée ---

def dice_loss(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    axes = [1, 2, 3]  # axes H, W, C  (format B H W C)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator  = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return tf.reduce_mean(1.0 - dice)


def combined_loss(y_true, y_pred, alpha=0.5):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce  = tf.reduce_mean(bce)
    dice = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dice

if __name__ == "__main__":
    batch_generate_masks_rimes("data/rimes_raw/DVD1", "data/rimes_masks")