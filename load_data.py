from image_process import *


# read test tfrecords
test_files = tf.train.match_filenames_once('./tfrecords/train.tfrecords')


# record parser
def parser(records):
    features = tf.parse_single_example(records, features={
                                   'image_data': tf.FixedLenFeature([], tf.string),
                                   'segmentation_data': tf.FixedLenFeature([], tf.string),
                                   })
    image, segmentation = features['image_data'], features['segmentation_data']
    image = tf.decode_raw(image, tf.float32)
    image = tf.reshape(image, [256, 256, 3])
    segmentation = tf.decode_raw(segmentation, tf.uint8)
    segmentation = tf.reshape(segmentation, [256, 256, 1])
    return image, segmentation


# create train batch
def get_train_dataset(batch_size, NUM_EPOCHS):
    shuffle_buffer = 10000
    # read train tfrecords
    dataset = tf.data.TFRecordDataset('./tfrecords/train.tfrecords')
    dataset = dataset.map(parser)
    dataset = dataset.map(lambda image_data, segmentation_data:
        preprocess_for_train(image_data, segmentation_data))
    dataset = dataset.map(lambda image, segmentation: (image, segmentation))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat()
    return dataset


# create train batch
def get_test_dataset(batch_size):
    dataset = tf.data.TFRecordDataset('./tfrecords/test.tfrecords')
    dataset = dataset.map(parser)
    dataset = dataset.map(lambda image, segmentation: (image, segmentation))
    dataset = dataset.batch(batch_size)
    return dataset

