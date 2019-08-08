import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from tfNLP.modeling.base_model import WordEmbedding, BiRnn, Classiffier

class BLSTMAttentionClassiffier():
    def __init__(self):
        with tf.variable_scope('test_model'):
            self.we = WordEmbedding()
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_type='gru')
            brnn_mean_outputs = tf.reduce_mean(self.brnn.outputs,axis=1)
            self.clf = Classiffier(inputs=brnn_mean_outputs, num_label=2, ffn_units_list=[30,30])
            for var in tf.trainable_variables(): print(var.name)   
 
    def test(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            fd={
              self.we.inputs: np.zeros([10,128]),
              self.brnn.sequence_length: [5 for i in range(10)],
            }
            outputs = sess.run([self.clf.outputs], feed_dict=fd)
            for o in outputs: print(o.shape)


if __name__=='__main__':
    m=BLSTMAttentionClassiffier()
    m.test()
