"""
unet.py
========
Small U-Net for ECG trace segmentation (binary mask: trace vs everything else).

Fully convolutional, so it accepts any image whose height and width are
divisible by 2**depth. Default 256 x 768 input, 3 channels (BGR / 255).
Loss = binary cross-entropy + Dice, because the trace covers only ~1-2% of
the pixels and plain BCE would happily predict "all background".
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras import layers, models


def _block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(height=256, width=768, base=16, depth=4):
    assert height % (2 ** depth) == 0 and width % (2 ** depth) == 0, \
        f"height/width must be divisible by {2 ** depth}"
    inp = layers.Input(shape=(height, width, 3), name="image")
    x, skips = inp, []
    for d in range(depth):
        x = _block(x, base * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    x = _block(x, base * 2 ** depth)
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = _block(x, base * 2 ** d)
    out = layers.Conv2D(1, 1, activation="sigmoid", name="trace_mask")(x)
    return models.Model(inp, out, name="ecg_unet")


def dice_coef(y_true, y_pred, smooth=1.0):
    yt = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    yp = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(yt * yp, axis=1)
    return tf.reduce_mean((2 * inter + smooth) /
                          (tf.reduce_sum(yt, 1) + tf.reduce_sum(yp, 1) + smooth))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coef(y_true, y_pred))


def iou_metric(y_true, y_pred, thr=0.5):
    yp = tf.cast(y_pred > thr, tf.float32)
    inter = tf.reduce_sum(y_true * yp, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, [1, 2, 3]) + tf.reduce_sum(yp, [1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1.0) / (union + 1.0))


def compile_unet(model, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=bce_dice_loss,
                  metrics=[dice_coef, iou_metric])
    return model


if __name__ == "__main__":
    m = build_unet()
    print("params:", f"{m.count_params():,}")
