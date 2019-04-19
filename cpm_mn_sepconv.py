import tensorflow as tf
import tensorflow.contrib.slim as slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_l2_regularizer_convb = tf.contrib.layers.l2_regularizer(0.004)

class PafNet:
    def __init__(self, inputs_x, use_bn=False, mask_paf=None, mask_hm=None, gt_hm=None, gt_paf=None, stage_num=6, hm_channel_num=19, paf_channel_num=38):
        self.inputs_x = inputs_x
        self.mask_paf = mask_paf
        self.mask_hm = mask_hm
        self.gt_hm = gt_hm
        self.gt_paf = gt_paf
        self.stage_num = stage_num
        self.paf_channel_num = paf_channel_num
        self.hm_channel_num = hm_channel_num
        self.use_bn = use_bn

    def add_layers(self, inputs):
        net = self.conv2(inputs=inputs, filters=256, padding='SAME', kernel_size=3, normalization=self.use_bn, name='cpm_1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn, name='cpm_2')
        # net = tf.layers.conv2d(inputs=inputs,
        #                              filters=256,
        #                              padding="same",
        #                              kernel_size=3,
        #                              activation="relu",
        #                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                              bias_initializer=tf.truncated_normal_initializer(stddev=0.1),)
        # net = tf.layers.conv2d(inputs=net,
        #                              filters=128,
        #                              padding="same",
        #                              kernel_size=3,
        #                              activation="relu",
        #                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                              bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        return net

    def stage_1(self, inputs, out_channel_num, name):
        # net = tf.layers.conv2d(inputs=inputs,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=3,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=3,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=3,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=512,
        #                        padding="same",
        #                        kernel_size=1,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=out_channel_num,
        #                        padding="same",
        #                        kernel_size=1,
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        net = self.conv2(inputs=inputs, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn, name=name+'_conv1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn, name=name+'_conv2')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn, name=name+'_conv3')
        net = self.conv2(inputs=net, filters=512, padding='SAME', kernel_size=1, normalization=self.use_bn, name=name+'_conv4')
        net = self.conv2(inputs=net, filters=out_channel_num, padding='SAME', kernel_size=1, act=False, normalization=self.use_bn, name=name+'_conv5')
        return net

    def stage_t(self, inputs, out_channel_num, name):
        # net = tf.layers.conv2d(inputs=inputs,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=7,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=7,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=7,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=7,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=7,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=128,
        #                        padding="same",
        #                        kernel_size=1,
        #                        activation="relu",
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # net = tf.layers.conv2d(inputs=net,
        #                        filters=out_channel_num,
        #                        padding="same",
        #                        kernel_size=1,
        #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                        bias_initializer=tf.truncated_normal_initializer(stddev=0.1))
        net = self.conv2(inputs=inputs, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn, name=name+'_conv1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn, name=name+'_conv2')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn, name=name+'_conv3')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn, name=name+'_conv4')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn, name=name+'_conv5')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=1, normalization=self.use_bn, name=name+'_conv6')
        net = self.conv2(inputs=net, filters=out_channel_num, padding='SAME', kernel_size=1, act=False, name=name+'_conv7')
        return net

    #     def conv2(self, inputs, filters, padding, kernel_size, name, act=True, normalization=False):
    #         channels_in = inputs[0, 0, 0, :].get_shape().as_list()[0]
    #         with tf.variable_scope(name) as scope:
    #             w = tf.get_variable('weights', shape=[kernel_size, kernel_size, channels_in, filters], trainable=True, initializer=tf.contrib.layers.xavier_initializer())
    #             b = tf.get_variable('biases', shape=[filters], trainable=True, initializer=tf.contrib.layers.xavier_initializer())
    #             conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding=padding)
    #             output = tf.nn.bias_add(conv, b)
    #             if normalization:
    #                 axis = list(range(len(output.get_shape()) - 1))
    #                 mean, variance = tf.nn.moments(conv, axes=axis)
    #                 scale = tf.Variable(tf.ones([filters]), name='scale')
    #                 beta = tf.Variable(tf.zeros([filters]), name='beta')
    #                 output = tf.nn.batch_normalization(output, mean, variance, offset=beta, scale=scale, variance_epsilon=0.0001)
    #             if act:
    #                 output = tf.nn.relu(output, name=scope.name)
    #         tf.summary.histogram('conv', conv)
    #         tf.summary.histogram('weights', w)
    #         tf.summary.histogram('biases', b)
    #         tf.summary.histogram('output', output)

    #         return output
    def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=True,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_h, k_w],
                                                  # activation_fn=common.activation_fn if relu else None,
                                                  activation_fn=None,
                                                  # normalizer_fn=slim.batch_norm,
                                                  weights_initializer=_init_xavier,
                                                  # weights_initializer=_init_norm,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding='SAME',
                                                  scope=name + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=_init_norm,
                                        biases_initializer=_init_zero if set_bias else None,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=True,
                                        weights_regularizer=None,
                                        scope=name + '_pointwise')

        return output
    def conv2(self, inputs, filters, padding, kernel_size, name, act=True, normalization=False):
        return self.separable_conv(inputs, kernel_size, kernel_size, filters, 1, name)

    def gen_net(self):
        paf_pre = []
        hm_pre = []
        with tf.variable_scope('openpose_layers'):
            with tf.variable_scope('cpm_layers'):
                # added_layers_out = self.add_layers(inputs=self.inputs_x)
                added_layers_out = self.inputs_x

            with tf.variable_scope('stage1'):
                paf_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.paf_channel_num, name='stage1_paf')
                hm_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.hm_channel_num, name='stage1_hm')
                paf_pre.append(paf_net)
                hm_pre.append(hm_net)
                net = tf.concat([hm_net, paf_net, added_layers_out], 3)

            with tf.variable_scope('staget'):
                for i in range(self.stage_num - 1):
                    hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num, name='stage%d_hm' % (i + 2))
                    paf_net = self.stage_t(inputs=net, out_channel_num=self.paf_channel_num, name='stage%d_paf' % (i + 2))
                    paf_pre.append(paf_net)
                    hm_pre.append(hm_net)
                    if i < self.stage_num - 2:
                        net = tf.concat([hm_net, paf_net, added_layers_out], 3)

        return hm_pre, paf_pre, added_layers_out

    # test code
    # def gen_net(self):
    #     paf_loss = []
    #     hm_loss = []
    #     paf_pre = []
    #     hm_pre = []
    #     with tf.variable_scope('add_layers'):
    #       added_layers_out = self.add_layers(inputs=self.inputs_x)
    #
    #     with tf.variable_scope('stage1'):
    #         # paf_net = self.stage_1(inputs=self.inputs_x, out_channel_num=self.paf_channel_num)
    #         hm_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.hm_channel_num)
    #         # paf_pre.append(paf_net)
    #         hm_pre.append(hm_net)
    #         # paf_loss.append(self.get_loss(paf_net, self.gt_paf, mask_type='paf'))
    #         hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
    #         net = tf.concat([hm_net, self.inputs_x], 3)
    #
    #     with tf.variable_scope('staget'):
    #         for i in range(self.stage_num - 1):
    #             with tf.name_scope("staget"):
    #                 hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num)
    #                 paf_net = self.stage_t(inputs=net, out_channel_num=self.paf_channel_num)
    #                 paf_pre.append(paf_net)
    #                 hm_pre.append(hm_net)
    #                 hm_loss.append(self.get_loss(hm_net, self.gt_hm, mask_type='hm'))
    #                 paf_loss.append(self.get_loss(paf_net, self.gt_paf, mask_type='paf'))
    #                 if i < self.stage_num - 2:
    #                     net = tf.concat([hm_net, self.inputs_x], 3)
    #
    #     # with tf.name_scope("loss"):
    #     #     total_loss = tf.reduce_sum(hm_loss)#  + tf.reduce_sum(paf_loss)
    #     # tf.summary.scalar("loss", total_loss)
    #     # tf.summary.image('hm_gt', self.gt_hm)
    #     return hm_pre, paf_pre

    # def get_loss(self, pre_y, gt_y, mask_type):
    #     if mask_type == 'paf':
    #         return tf.reduce_mean(tf.reduce_sum(tf.square(gt_y - pre_y) * self.mask_paf, axis=[1, 2, 3]))
    #     if mask_type == 'hm':
    #         # return tf.reduce_mean(tf.reduce_sum(tf.square(gt_y - pre_y) * self.mask_hm, axis=[1, 2, 3]))
    #         # return tf.losses.sigmoid_cross_entropy(gt_y, pre_y)
    #         return tf.reduce_sum(tf.nn.l2_loss(gt_y - pre_y,))
    #             #(pre_y, gt_y)
