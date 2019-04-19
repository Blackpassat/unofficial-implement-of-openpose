import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
from tensorflow.contrib import slim
import vgg
import mobilenet_v2
import mobilenet
from cpm import PafNet
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator
import pdb
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import os
import json

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2])
    return keypoints

def round_int(val):
    return int(round(val))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing code for Openpose using Tensorflow')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/2019-3-23-19-39-4/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/mobilenet/mobilenet_v2_1.0_96.ckpt')
    parser.add_argument('--train_mobilenet', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--image_path', type=str, default='./COCO/images/val2017/')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, 619, 654, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    layers = {}
    name = ""
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(img_normalized)
        for k, tensor in sorted(list(endpoints.items()), key=lambda x: x[0]):
            layers['%s%s' % (name, k)] = tensor
            print(k, tensor.shape)
    def upsample(input, target):
        return tf.image.resize_bilinear(input, tf.constant([target.shape[1].value, target.shape[2].value]), align_corners=False)
    
    mobilenet_feature = tf.concat([layers['layer_7/output'], upsample(layers['layer_14/output'], layers['layer_7/output'])], 3)
    
    # get net graph
    logger.info('initializing model...')
    # net = PafNet(inputs_x=vgg_outputs, use_bn=args.use_bn)
    # hm_pre, cpm_pre, added_layers_out = net.gen_net()
    net = PafNet(inputs_x=mobilenet_feature, stage_num=6, hm_channel_num=19, use_bn=args.use_bn)
    hm_pre, paf_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(paf_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_mobilenet:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2'), name='mobilenet_restorer')
    saver = tf.train.Saver(trainable_var_list)
    logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring mobilenet weights...')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        saver.restore(sess, "./checkpoints/train/2019-3-23-19-39-4/model-12000")
        #saver.restore(sess, args.checkpoint_path + "model-8.ckpt")
        logger.info('initialization done')

        ## variables for file namess etc
        result = []
        write_json = "result.json"

        image_dir = args.image_path
        coco_json_file = './COCO/annotations/person_keypoints_val2017.json'
        cocoGt = COCO(coco_json_file)
        catIds = cocoGt.getCatIds(catNms=['person'])
        keys = cocoGt.getImgIds(catIds=catIds)
        tqdm_keys = tqdm(keys)
        for i, k in enumerate(tqdm_keys):
            img_meta = cocoGt.loadImgs(k)[0]    
            img_idx = img_meta['id']

            img_name = os.path.join(image_dir, img_meta['file_name'])
            input_image = common.read_imgfile(img_name, None, None)

            size = [input_image.shape[0], input_image.shape[1]]
            if input_image is None:
                logger.error('Image can not be read, path=%s' % input_image)
                sys.exit(-1)        
            # h = int(654 * (size[0] / size[1]))
            img = np.array(cv2.resize(input_image, (654, 619)))
            # cv2.imshow('ini', img)
            img = img[np.newaxis, :]
            peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up], feed_dict={raw_img: img, img_size: size})
            # cv2.imshow('in', vectormap[0, :, :, 0])
            humans = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])

            for human in humans:
                item = {
                    'image_id': img_idx,
                    'category_id': 1,
                    'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
                    'score': human.score
                }
                result.append(item)

        fp = open(write_json, 'w')
        json.dump(result, fp)
        fp.close()

        # image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
        # # cv2.imshow(' ', image)
        # cv2.imwrite("result.png", image)
        # cv2.waitKey(0)
        
        ### call COCO apis for json file evaluation
        cocoDt = cocoGt.loadRes(write_json)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = keys
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(''.join(["%11.4f |" % x for x in cocoEval.stats]))

        pred = json.load(open(write_json, 'r'))
        pdb.set_trace()
