import numpy as np
from load_data import *


class Model(object):
  
    def __init__(self, out_channels=21,  batch_size=8):
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.height = 256
        self.width = 256
        self.NUM_EPOCHS = 80
        self.input_image = None
        self.input_segmentation = None
        self.prediction = None
        self.input_segmentation = None
        self.loss = None
        self.loss_mean = None
        self.loss_all = None
        self.pixel_accuracy = None
        self.train_step = None

    def create_model(self):
        # input
        with tf.name_scope('input'):
            self.input_image = tf.placeholder(
                dtype=tf.float32, shape=[self.batch_size, 256, 256, 3],
                name='input_images'
            )
            self.input_segmentation = tf.placeholder(
                dtype=tf.int32, shape=[self.batch_size, 256, 256],
                name='input_segmentations'
            )
        # encoder
        # input size [batch_size, w, h, 3]
        # [ w, h, 64]
        conv1_1 = self.conv(self.input_image, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        # [ w, h, 64]
        conv1_2 = self.conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        # [ w/2, h/2, 64]
        pool1 = self.max_pool(conv1_2, 2, 2, 2, 2, padding='SAME', name='pool1')
        # [ w/2, h/2, 128]
        conv2_1 = self.conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        # [ w/2, h/2, 128]
        conv2_2 = self.conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        # [ w/4, h/4, 128]
        pool2 = self.max_pool(conv2_2, 2, 2, 2, 2, padding='SAME', name='pool2')
        # [ w/4, h/4, 256]
        conv3_1 = self.conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        # [ w/4, h/4, 256]
        conv3_2 = self.conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        # [ w/4, h/4, 256]
        conv3_3 = self.conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        # [ w/8, h/8, 256]
        pool3 = self.max_pool(conv3_3, 2, 2, 2, 2, padding='SAME', name='pool3')
        # [ w/8, h/8, 512]
        conv4_1 = self.conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        # [ w/8, h/8, 512]
        conv4_2 = self.conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        # [ w/8, h/8, 512]
        conv4_3 = self.conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        # [ w/16, h/16, 512]
        pool4 = self.max_pool(conv4_3, 2, 2, 2, 2, padding='SAME', name='pool4')
        # [ w/16, h/16, 512]
        conv5_1 = self.conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        # [ w/16, h/16, 512]
        conv5_2 = self.conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        # [ w/16, h/16, 512]
        conv5_3 = self.conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        # [ w/32, h/32, 512]
        pool5 = self.max_pool(conv5_3, 2, 2, 2, 2, padding='SAME', name='pool5')
        # [ w/32, h/32, 512]
        center = self.conv(pool5, 3, 3, 512, 1, 1, padding='SAME', name='center')

        # decoder
        # [~, w/16, h/16, 512]
        deconv1 = self.deconv(center, 3, 3, 512, 2, 2,
                              output_shape=[self.batch_size, self.height/16, self.width/16, 512],
                              padding='SAME', name='deconv1')
        # [~, w/16, h/16, 512]
        norm1 = self.norm_rescale(pool4, 512, 'norm1')
        # [~, w/16, h/16, 1024]
        concat1 = tf.concat([deconv1, norm1], axis=3)
        # [~, w/16, h/16, 512]
        conv6_1 = self.conv(concat1, 3, 3, 512, 1, 1, padding='SAME', name='conv6_1')
        # [~, w/16, h/16, 256]
        conv6_2 = self.conv(conv6_1, 3, 3, 256, 1, 1, padding='SAME', name='conv6_2')
        # [~, w/8, h/8, 256]
        deconv2 = self.deconv(conv6_2, 3, 3, 256, 2, 2,
                              output_shape=[self.batch_size, self.height/8, self.width/8, 256],
                              padding='SAME', name='deconv2')
        # [~, w/8, h/8, 256]
        norm2 = self.norm_rescale(pool3, 256, 'norm2')
        # [~, w/8, h/8, 512]
        concat2 = tf.concat([deconv2, norm2], axis=3)
        # [~, w/8, h/8, 256]
        conv7_1 = self.conv(concat2, 3, 3, 256, 1, 1, padding='SAME', name='conv7_1')
        # [~, w/8, h/8, 128]
        conv7_2 = self.conv(conv7_1, 3, 3, 128, 1, 1, padding='SAME', name='conv7_2')
        # [~, w/4, h/4, 128]
        deconv3 = self.deconv(conv7_2, 3, 3, 128, 2, 2,
                              output_shape=[self.batch_size, self.height/4, self.width/4, 128],
                              padding='SAME', name='deconv3')
        # [~, w/4, h/4, 128]
        norm3 = self.norm_rescale(pool2, 128, 'norm3')
        # [~, w/4, h/4, 256]
        concat3 = tf.concat([deconv3, norm3], axis=3)
        # [~, w/4, h/4, 128]
        conv8_1 = self.conv(concat3, 3, 3, 128, 1, 1, padding='SAME', name='conv8_1')
        # [~, w/4, h/4, 64]
        conv8_2 = self.conv(conv8_1, 3, 3, 64, 1, 1, padding='SAME', name='conv8_2')
        # [~, w/2, h/2, 64]
        deconv4 = self.deconv(conv8_2, 3, 3, 64, 2, 2,
                              output_shape=[self.batch_size, self.height/2, self.width/2, 64],
                              padding='SAME', name='deconv4')
        # [~, w/2, h/2, 64]
        norm4 = self.norm_rescale(pool1, 64, 'norm4')
        # [~, w/2, h/2, 128]
        concat4 = tf.concat([deconv4, norm4], axis=3)
        # [~, w/2, h/2, 64]
        conv9_1 = self.conv(concat4, 3, 3, 64, 1, 1, padding='SAME', name='conv9_1')
        # [~, w, h, 64]
        deconv5 = self.deconv(conv9_1, 3, 3, self.out_channels, 2, 2,
                              output_shape=[self.batch_size, self.height, self.width, 21],
                              padding='SAME', name='deconv5')
        # [~, w, h, out_channels(<64)]
        self.prediction = self.conv(deconv5, 1, 1, self.out_channels, 1, 1,
                                    padding='SAME', name='conv10_1')
        # self.prediction = self.prediction + tf.constant(1e-5, shape=(16, 448, 448, 21))
        # loss function
        with tf.name_scope('softmax_loss'):
            # not using one-hot
            self.loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_segmentation,
                                                               logits=self.prediction,
                                                               name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # pixel accuracy
        with tf.name_scope('softmax_loss'):
            output_mask = tf.cast(tf.arg_max(tf.nn.softmax(self.prediction), 3), tf.int32)
            self.pixel_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(output_mask, self.input_segmentation), tf.float32))

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)


    # load train data and train model
    def train(self):
        train_dataset = get_train_dataset(self.batch_size, self.NUM_EPOCHS)
        # use iterator to travel dataset
        iterator = train_dataset.make_initializable_iterator()
        # iterator = train_dataset.make_one_shot_iterator()
        image_batch, segmentation_batch = iterator.get_next()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator.initializer)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while epoch < 1000:
                    # Run training steps or whatever
                    # sess.run([next_batch])
                    # print sess.run([next_batch['image_data']])
                    # images = batch['image_data']
                    # segmentations = batch['segmentation_data'] 
                    images = sess.run([image_batch])
                    segmentations = sess.run([segmentation_batch])
                    images = images[0]
                    segmentations = segmentations[0]
                    segmentations = segmentations.reshape([self.batch_size, 256, 256])
                    segmentations = segmentations.astype('int')
                    images = images.astype('float') 
                    loss = self.loss_mean.eval(feed_dict = {self.input_image: images, self.input_segmentation:  segmentations})
                    pixel_accuracy = self.pixel_accuracy.eval(feed_dict = {self.input_image: images, self.input_segmentation:  segmentations})

                    #print self.prediction
                    if epoch % 10 == 0:
                        print epoch, loss, pixel_accuracy
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: images, self.input_segmentation: segmentations}
                    )
                    if epoch in [400, 600, 800]:
                        all_parameters_saver.save(sess=sess, save_path='./checkpoint/moddel-' + str(epoch))
                    epoch += 1
            finally:
                # When done, ask the threads to stop.
                all_parameters_saver.save(sess=sess, save_path='./checkpoint/model.ckpt')
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print("Done training")

    @staticmethod
    def conv(x, filter_height, filter_width, num_filters,
             stride_y, stride_x, padding, name):
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            kernel = tf.get_variable('kernel',
                                     shape=[filter_height, filter_width, input_channels, num_filters],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(x, kernel,
                                strides=[1, stride_y, stride_x, 1], padding=padding)
            bias_init_val = tf.constant(0.0, shape=[num_filters], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            z = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z, name=scope.name)
            return activation

    @staticmethod
    def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    @staticmethod
    def deconv(x, filter_height, filter_width, num_filters,
               stride_y, stride_x, output_shape, name, padding='SAME'):
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            kernel = tf.get_variable('kernel',
                                     shape=[filter_height, filter_width, num_filters, input_channels],
                                     dtype=tf.float32)
            dconv = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, stride_y, stride_x, 1], padding)
            biases = tf.get_variable('biases', shape=[num_filters])
            z = tf.nn.bias_add(dconv, biases)
            activation = tf.nn.relu(z, name=scope.name)
            return activation

    @staticmethod
    def norm_rescale(x, channels, name):
        with tf.variable_scope(name) as scope:
            scale = tf.get_variable('scale', shape=[channels], trainable=True, dtype=tf.float32)
            return scale * tf.nn.l2_normalize(x, dim=[1, 2])

model = Model()
model.create_model()
model.train()
