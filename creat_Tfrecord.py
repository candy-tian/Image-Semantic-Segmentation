import tensorflow as tf

train_image_list = []
test_image_list = []
train_file = open('./data/train.txt')
for line in train_file:
    if line != '':
        train_image_list.append(line.strip()[:])
test_file = open('./data/val.txt')
for line in test_file:
    if line != '':
        test_image_list.append(line.strip()[:])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

image_dir = './data/JPEGImages/'
segmentation_dir = './data/labels/'


def write_data_2_tfrecords(file_list, record_path):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.python_io.TFRecordWriter(record_path)
        for filename in file_list:
            image_data = tf.gfile.FastGFile(image_dir + filename + '.jpg', 'r').read()
            segmentation_data = tf.gfile.FastGFile(segmentation_dir + filename + '.png', 'r').read()
            image_data = tf.image.decode_jpeg(image_data)
            segmentation_data = tf.image.decode_png(segmentation_data)
            image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
            # segmentation_data = tf.image.convert_image_dtype(segmentation_data, dtype=tf.int32)
            image_data = tf.image.resize_images(image_data, [256, 256], method=0)
            segmentation_data = tf.image.resize_images(segmentation_data, [256, 256], method=1)
            image_data = sess.run(image_data)
            segmentation_data = sess.run(segmentation_data)
            image_data = image_data.tostring()
            segmentation_data = segmentation_data.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_data': _bytes_feature(image_data),
                'segmentation_data': _bytes_feature(segmentation_data),
            }))
            writer.write(example.SerializeToString())
        writer.close()
        coord.join(threads)

write_data_2_tfrecords(train_image_list, './tfrecords/train.tfrecords')
write_data_2_tfrecords(test_image_list, './tfrecords/test.tfrecords')

