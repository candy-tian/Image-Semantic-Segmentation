import tensorflow as tf
import numpy as np


def randomly_distort_color(image, color_ordering):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def randomly_fip_image_and_segmentation(image, segmentation, random_var):
    if random_var == 0:
        image = tf.image.flip_up_down(image)
        segmentation = tf.image.flip_up_down(segmentation)
    elif random_var == 1:
        image = tf.image.flip_left_right(image)
        segmentation = tf.image.flip_left_right(segmentation)
    elif random_var == 2:
        image = tf.image.transpose_image(image)
        segmentation = tf.image.transpose_image(segmentation)
    return image, segmentation


def preprocess_for_train(image, segmentation):
    # convert dtype to float32
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    segmentation = tf.image.convert_image_dtype(segmentation, dtype=tf.float32)
    # randomly distort color of image
    image = randomly_distort_color(image, np.random.choice(4))
    # randomly flip image and segmentation
    image, segmentation = randomly_fip_image_and_segmentation(image, segmentation, np.random.choice(3))
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    segmentation = tf.image.convert_image_dtype(segmentation, dtype=tf.uint8)
    return image, segmentation

# def preprocess_for_test(image, segmentation, height, width):
#     return
