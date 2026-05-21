# model.py
from tensorflow import keras
from keras import layers


def double_conv(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x, filters):
    skip = double_conv(x, filters)
    x    = layers.MaxPooling2D(2)(skip)
    return skip, x


def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    # Recadrage si dimensions impaires
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = layers.Resizing(skip.shape[1], skip.shape[2])(x)
    x = layers.Concatenate()([x, skip])
    x = double_conv(x, filters)
    return x


def build_unet(input_shape=(512, 512, 1), base_features=64):
    inputs = keras.Input(shape=input_shape)

    # Encodeur
    s1, x = encoder_block(inputs, base_features)       # 256×256
    s2, x = encoder_block(x, base_features * 2)        # 128×128
    s3, x = encoder_block(x, base_features * 4)        # 64×64
    s4, x = encoder_block(x, base_features * 8)        # 32×32

    # Bottleneck
    x = double_conv(x, base_features * 16)             # 32×32, 1024 filtres

    # Décodeur
    x = decoder_block(x, s4, base_features * 8)        # 64×64
    x = decoder_block(x, s3, base_features * 4)        # 128×128
    x = decoder_block(x, s2, base_features * 2)        # 256×256
    x = decoder_block(x, s1, base_features)            # 512×512

    # Sortie
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="UNet")


if __name__ == "__main__":
    model = build_unet()
    model.summary()