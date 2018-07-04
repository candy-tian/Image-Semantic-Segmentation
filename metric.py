from load_data import *
import tensorflow as tf
import numpy as np


def model_val(model_index):
    ckpt_path = './checkpoint/moddel-' + str(model_index)
    graph_path = ckpt_path + '.meta'
    loss_all = []
    accuracy_all = []
    test_dataset = get_test_dataset(8)
    iterator = test_dataset.make_initializable_iterator()
    image_batch, segmentation_batch = iterator.get_next()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(graph_path)
        new_saver.restore(sess, ckpt_path)
        sess.run(iterator.initializer)
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input/input_images:0")
        input_y = graph.get_tensor_by_name("input/input_segmentations:0")
        batch_loss = graph.get_tensor_by_name("softmax_loss/loss/loss:0")
        prediction = graph.get_tensor_by_name("conv10_1/conv10_1:0")
        loss_mean = tf.reduce_mean(batch_loss)
        output_mask = tf.cast(tf.arg_max(tf.nn.softmax(prediction), 3), tf.int32)
        pixel_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(output_mask, input_y), tf.float32))
        while True:
            images = sess.run([image_batch])
            segmentations = sess.run([segmentation_batch])
            images = images[0]
            segmentations = np.squeeze(segmentations[0])
            if images.shape[0] != 8 or segmentations.shape[0] != 8:
                break
            segmentations = segmentations.astype('int')
            images = images.astype('float')
            feed_dict = {input_x: images, input_y: segmentations}
            loss = sess.run(loss_mean, feed_dict)
            accuracy = sess.run(pixel_accuracy, feed_dict)
            loss_all.append(loss)
            accuracy_all.append(accuracy)
        loss_all = np.array(loss_all)
        accuracy_all = np.array(accuracy_all)
        print(loss_all.mean(), accuracy_all.mean())

index_list = [400, 600, 800, 1000, 2000, 3000, 4000]
for model_index in index_list:
    model_val(model_index)
