import os,sys,re,time,shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell

class WordEmbedding():
    def __init__(self, num_step=128, dict_size=20000, word_vec_size=100,name='wordembedding'):
        with tf.name_scope(name):
            self.inputs=tf.placeholder(tf.int32,[None, num_step])
            self.wordEmbedding = tf.Variable(tf.random_normal([dict_size, word_vec_size], stddev=0.1))
            self.outputs = tf.nn.embedding_lookup(self.wordEmbedding, self.inputs)
            self.saver = tf.train.Saver([var for var in tf.trainable_variables() if name in var.name])

class BiRnn():
    def __init__(self, inputs, rnn_type = 'lstm', rnn_size_list=[10,10], is_train=False, rnn_keep_prob=0.99,name='birnn'):
        with tf.variable_scope(name):
            self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
            if rnn_type=='lstm':
                RnnCell=rnn_cell.LSTMCell
            elif rnn_type=='gru':
                RnnCell=rnn_cell.GRUCell
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([RnnCell(rz) for rz in rnn_size_list])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([RnnCell(rz) for rz in rnn_size_list])
            keep_prob = rnn_keep_prob if is_train else 1.0
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            #调用双向RNN，提取词向量序列(文本)的特征
            outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.sequence_length, dtype=tf.float32)
            #对两个LSTM输出的序列进行相加，成为一个序列
            self.outputs = tf.concat(outputs,axis=2)
            self.saver = tf.train.Saver([var for var in tf.trainable_variables() if name in var.name])

class Dropout():
    def __init__(self, inputs, keep_prob=0.5, name='dropout'):
        with tf.variable_scope(name):
            self.outputs = tf.nn.dropout(inputs, keep_prob)

class Attention():
    def __init__(self, inputs, name='tfnlp_attention'):
        with tf.variable_scope(name):
            raw_inputs_shape = inputs.get_shape().as_list()
            W = tf.Variable(tf.random_normal([raw_inputs_shape[2]], stddev=0.1))  
            M = tf.tanh(inputs)
            _M = tf.matmul(tf.reshape(M, [-1, raw_inputs_shape[2]]), tf.reshape(W, [-1, 1]))  
            _M = tf.reshape(_M, [-1, raw_inputs_shape[1]])  
            self.alpha = tf.nn.softmax(_M) #alpah因子，使得模型可以解释，根据alpha可以知道哪些词汇对最终的分类结果影响更大。  
            #利用权重因子alpha对LSTM输出序列进行加权求和，注意力越高的输出对于最终结果的影响越大。  
            r = tf.matmul(tf.transpose(inputs, [0, 2, 1]), tf.reshape(self.alpha, [-1, raw_inputs_shape[1], 1]))  
            self.outputs = tf.nn.tanh(tf.reshape(r, [-1, raw_inputs_shape[2]]))  

class CRF():
    def __init__(self, inputs, num_tags=2, sequence_length=None,name='crf'):
        with tf.variable_scope(name):
            raw_inputs_shape = inputs.get_shape().as_list()
            self.exp_tags = tf.placeholder(tf.int32, [None, raw_inputs_shape[1]])
            inputs=tf.reshape(inputs,[-1, raw_inputs_shape[2]])
            inputs = tf.layers.dense(inputs, units=num_tags, activation=tf.nn.tanh)
            inputs = tf.reshape(inputs,[-1,raw_inputs_shape[1],num_tags])
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs, self.exp_tags, sequence_length)
            self.loss = -tf.reduce_mean(log_likelihood)
            self.outputs, self.best_score = tf.contrib.crf.crf_decode(inputs,self.transition_params,sequence_length)
            yeqd = tf.equal(self.exp_tags, self.outputs)
            sum_sl = tf.to_float(tf.reduce_sum(sequence_length))
            self.acc = (tf.reduce_sum(tf.to_float(yeqd)) - tf.reduce_sum(tf.ones_like(tf.to_float(self.exp_tags))) + sum_sl) / sum_sl
            self.saver = tf.train.Saver([var for var in tf.trainable_variables() if name in var.name])
            #tf.summary.scalar('acc',self.acc)
            #tf.summary.scalar('loss',self.loss)

class Classiffier():
    def __init__(self, inputs, num_label, ffn_units_list=[],name='classiffier'):
        with tf.variable_scope(name):
            self.exp_label = tf.placeholder(tf.int32, [None])
            exp_prob = tf.one_hot(self.exp_label, depth=num_label)
            for units in ffn_units_list:
                inputs = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
            self.energy = tf.layers.dense(inputs=inputs, units=num_label, activation=tf.nn.tanh)
            self.outputs = tf.nn.softmax(self.energy)
            self.loss = -tf.reduce_mean(tf.reduce_sum(exp_prob*tf.log(self.outputs),axis=1))
            self.predict_label = tf.to_int32(tf.argmax(self.outputs,axis=1))
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.predict_label, self.exp_label)))
            self.all_label = tf.concat([tf.reshape(self.exp_label,[-1,1]),tf.reshape(self.predict_label,[-1,1])],axis=1)
            self.saver = tf.train.Saver([var for var in tf.trainable_variables() if name in var.name])
            #tf.summary.scalar('acc',self.acc)
            #tf.summary.scalar('loss',self.loss)
            
class Regression():
    def __init__(self, inputs, num_out, ffn_units_list=[],name='regression'):
        with tf.variable_scope(name):
            self.exp_out = tf.placeholder(tf.float32, [None, num_out])
            for units in ffn_units_list:
                inputs = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
            self.outputs = tf.layers.dense(inputs=inputs, units=num_out, activation=tf.nn.tanh)
            self.loss = -tf.reduce_mean((self.outputs-self.exp_out)**2)
            self.stderr = self.loss**0.5
            self.saver = tf.train.Saver([var for var in tf.trainable_variables() if name in var.name])

class SelfAttention():
    pass

class Bert():
     pass

class TestModel():
    def __init__(self):
        with tf.variable_scope('test_model'):
            self.we = WordEmbedding()
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_type='gru')
            brnn_mean_outputs = tf.reduce_mean(self.brnn.outputs,axis=1)
            #分类
            self.clf = Classiffier(inputs=brnn_mean_outputs, num_label=2, ffn_units_list=[20,10,5])
            #回归
            self.rgs = Regression(inputs=brnn_mean_outputs, num_out=2, ffn_units_list=[20,10,5])
            #NER
            self.crf = CRF(inputs=self.brnn.outputs,sequence_length=self.brnn.sequence_length)
            for var in tf.trainable_variables(): print(var.name)   
 
    def test(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            fd={
              self.we.inputs: np.zeros([10,128]),
              self.brnn.sequence_length: [5 for i in range(10)],
            }
            outputs = sess.run([self.crf.outputs,self.clf.outputs,self.rgs.outputs], feed_dict=fd)
            for o in outputs: print(o.shape)
            assert(outputs[0].shape==(10,128))
            assert(outputs[1].shape==(10,2))
            assert(outputs[2].shape==(10,2))

if __name__=='__main__':
    m=TestModel()
    m.test()
