import tensorflow as tf


def _resize_(image, mask):
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.image.resize(mask, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask


def _random_crop_(image, mask):
    concat = tf.concat([image, mask], axis=2)
    concat = tf.image.random_crop(concat, size=[128, 128, 3 + 3])
    image, mask = tf.split(concat, [3, 3], axis=2)
    return image, mask


def _flip_left_right_(image, mask):
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    return image, mask


def _flip_up_down_(image, mask):
    image = tf.image.random_flip_up_down(image)
    mask = tf.image.random_flip_up_down(mask)
    return image, mask


def _adjust_brightness_(image, mask):
    image = tf.image.adjust_brightness(image, 0.2)
    return image, mask


def _adjust_contrast_(image, mask):
    image = tf.image.adjust_contrast(image, 3)
    return image, mask


def _adjust_saturation(image, mask):
    image = tf.image.adjust_saturation(image, 3)
    return image, mask


def _normalize_(image, mask):
    image = image / 255
    mask = mask / 255
    return image, mask


def train_augmentation(image, mask):
    image, mask = _resize_(image, mask)
    # image, mask = _random_crop_(image, mask)
    image, mask = _flip_left_right_(image, mask)
    image, mask = _flip_up_down_(image, mask)
    image, mask = _adjust_brightness_(image, mask)
    # image, mask = _adjust_contrast_(image, mask)
    # image, mask = _adjust_saturation(image, mask)
    image, mask = _normalize_(image, mask)
    return image, mask


def validation_augmentation(image, mask):
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.image.resize(mask, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image, mask = _normalize_(image, mask)
    return image, mask
