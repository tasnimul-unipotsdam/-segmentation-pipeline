import tensorflow as tf
from pipeline.dataloader import DataLoader
from pipeline.read import train_image_path, train_mask_path, validation_image_path, validation_mask_path
import os
import random
random.seed(1001)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

train_dataset = DataLoader(train_image_path, train_mask_path, training=True).load_data()
validation_dataset = DataLoader(validation_image_path, validation_mask_path, training=False).load_data()

inputs = tf.keras.Input(shape=(256, 256, 3))
conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
    inputs)
conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
    conv1)
pool1 = tf.keras.layers.MaxPool2D()(conv1)

conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
    pool1)
conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
    conv2)
pool2 = tf.keras.layers.MaxPool2D()(conv2)

conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    pool2)
conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    conv3)
pool3 = tf.keras.layers.MaxPool2D()(conv3)

conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(
    pool3)
conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(
    conv4)

conc5 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256,
                                                                     (2, 2),
                                                                     strides=(
                                                                     2, 2),
                                                                     padding="same")(
    conv4), conv3], axis=3)
conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    conc5)
conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    conv5)

conc6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128,
                                                                     (2, 2),
                                                                     strides=(
                                                                     2, 2),
                                                                     padding="same")(
    conv5), conv2], axis=3)
conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
    conc6)
conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
    conv6)

conc7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64,
                                                                     (2, 2),
                                                                     strides=(
                                                                     2, 2),
                                                                     padding="same")(
    conv6), conv1], axis=3)
conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
    conc7)
conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
    conv7)

conv8 = tf.keras.layers.Conv2D(150, (1, 1), activation="softmax")(conv7)

model = tf.keras.Model(inputs, conv8, name='U-Net_model')

# model.summary()

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy', MeanIoU(name='iou', num_classes=150)])

EPOCHS = 1

STEPS_PER_EPOCH = 20210 // 4
VALIDATION_STEPS = 2000 // 4
model_history = model.fit(train_dataset, verbose=2,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=validation_dataset)