import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools
from queues import *
from generator import *       
from utils_multistep_lr import *
import math
import numpy as np
class EWNet(Model):
    def conv2d(self,inputs,filters=16,kernel_size=3,strides=1,padding='SAME',activation=None,
                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                       kernel_regularizer=layers.l2_regularizer(2e-4),
                       use_bias=False,name="conv"):
      return tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last",activation=activation,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias,name=name)
    def _build_model(self, inputs):
        if self.data_format == 'NCHW': 
            reduction_axis = [2,3]
            concat_axis = 1
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:          
            reduction_axis = [1,2]
            concat_axis = 3
            _inputs = tf.cast(inputs, tf.float32)
        self.inputImage = _inputs
        with arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format),\
            arg_scope([layers.max_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):     
          with tf.variable_scope('Preprocess'):                  
              conv = self.conv2d(_inputs,filters=12,name='preprocess',kernel_size=3)
              actv = tf.nn.relu(layers.batch_norm(conv))
          with tf.variable_scope('Layer2'): 
              conv1=self.conv2d(actv,filters=12,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=24,strides=2,name="conv2")
              actv2=layers.batch_norm(conv2)
              conv3=self.conv2d(actv,filters=24,strides=2,name="conv3")
              actv3=layers.batch_norm(conv3)
              res=tf.add_n([actv3,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer3'):  
              conv1=self.conv2d(res,filters=24,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=24,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer4'): 
              conv1=self.conv2d(res,filters=24,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=48,strides=2,name="conv2")
              actv2=layers.batch_norm(conv2)
              conv3=self.conv2d(res,filters=48,strides=2,name="conv3")
              actv3=layers.batch_norm(conv3)
              res=tf.add_n([actv3,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer5'): 
              conv1=self.conv2d(res,filters=48,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=48,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer6'): 
              conv1=self.conv2d(res,filters=48,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=96,strides=2,name="conv2")
              actv2=layers.batch_norm(conv2)
              conv3=self.conv2d(res,filters=96,strides=2,name="conv3")
              actv3=layers.batch_norm(conv3)
              res=tf.add_n([actv3,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer7'): 
              conv1=self.conv2d(res,filters=96,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=96,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)  
          with tf.variable_scope('Layer8'): 
              conv1=self.conv2d(res,filters=96,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=192,strides=2,name="conv2")
              actv2=layers.batch_norm(conv2)
              conv3=self.conv2d(res,filters=192,strides=2,name="conv3")
              actv3=layers.batch_norm(conv3)
              res=tf.add_n([actv3,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer9'): 
              conv1=self.conv2d(res,filters=192,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=192,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer10'): 
              conv1=self.conv2d(res,filters=192,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=384,strides=2,name="conv2")
              actv2=layers.batch_norm(conv2)
              conv3=self.conv2d(res,filters=384,strides=2,name="conv3")
              actv3=layers.batch_norm(conv3)
              res=tf.add_n([actv3,actv2])
              res=tf.nn.relu(res)        
          with tf.variable_scope('Layer11'): 
              deconv1_shape=[3,3,192,384]
              initial = tf.truncated_normal(deconv1_shape, stddev=0.01)
              deconv1 = tf.get_variable(name="deconv1",initializer=initial)
              shape = res.get_shape().as_list()
              H = shape[1]*2
              W = shape[2]*2
              output_shape=[32,H,W,192]
              strides=[1,2,2,1]
              res=tf.nn.conv2d_transpose(res, deconv1, output_shape, strides, padding='SAME', data_format='NHWC', name='None')
          with tf.variable_scope('Layer12'): 
              deconv2_shape=[3,3,32,192]
              initial2 = tf.truncated_normal(deconv2_shape, stddev=0.01)
              deconv2 = tf.get_variable(name="deconv2",initializer=initial2)
              shape = res.get_shape().as_list()
              H = shape[1]*4
              W = shape[2]*4
              output_shape=[32,H,W,32]
              strides=[1,4,4,1]
              res=tf.nn.conv2d_transpose(res, deconv2, output_shape, strides, padding='SAME', data_format='NHWC', name='None')
          with tf.variable_scope('Layer13'):
              deconv2_shape=[3,3,2,32]
              initial2 = tf.truncated_normal(deconv2_shape, stddev=0.01)
              deconv2 = tf.get_variable(name="deconv2",initializer=initial2)
              shape = res.get_shape().as_list()
              H = shape[1]*4
              W = shape[2]*4
              output_shape=[32,H,W,2]
              strides=[1,4,4,1]
              res=tf.nn.conv2d_transpose(res, deconv2, output_shape, strides, padding='SAME', data_format='NHWC', name='None')
          with tf.variable_scope('Layer14'):
              soft_max = tf.nn.softmax(res,dim=3,name="feature_pre")
              x=tf.reduce_mean(soft_max, axis=[1,2],keep_dims=True)
              logits=layers.flatten(x)
        self.outputs = logits
