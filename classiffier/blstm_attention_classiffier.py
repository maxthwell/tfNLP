import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from tfNLP.modeling.base_model import WordEmbedding, BiRnn, Classiffier, Attention
from tfNLP.data_processor.processor import ClassiffierDataProcessor as CDP
from tfNLP.classiffier.evaluator import ClassiffierModelEvaluator

class BLSTMAttentionClassiffier():
    def __init__(self, cdp, model_path=None):
        self.model_path=model_path
        self.metrics={}
        self.cv_sampler = None
        self.cdp = cdp
        with tf.variable_scope('blstm_attention_classiffier'):
            self.we = WordEmbedding(num_steps=cdp.num_steps,dict_size=self.cdp.num_words,word_vec_size=30)
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
            attention = Attention(inputs=self.brnn.outputs)
            self.clf = Classiffier(inputs=attention.outputs, num_label=self.cdp.num_labels, ffn_units_list=[30,30])
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.clf.loss, global_step=self.clf.global_step)
            self.saver = tf.train.Saver()
            for var in tf.trainable_variables(): print(var.name)   


    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self, sess):
        if not self.cv_sampler: self.cv_sampler = self.cdp.batch_sample(batch_size=10000, work_type='cv')
        S,L,X,Y = next(self.cv_sampler)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.brnn.sequence_length: np.array(L,dtype=np.int32),
          self.clf.exp_label: np.array(Y,dtype=np.int32),
        }
        t0=time.time()
        step,loss,acc,all_label = sess.run([self.clf.global_step, self.clf.loss, self.clf.acc, self.clf.all_label],feed_dict=fd)
        t1=time.time()
        print('----------------- global_step: ',step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        eva = ClassiffierModelEvaluator(num_labels=self.cdp.num_labels,label_name_list=self.cdp.label_list)
        eva.batch_add(all_label)
        eva.update_quota()
        print('cross-validation ---------- loss: %s, acc:%s'%(loss,acc))
        print(eva)
        if 'loss' not in self.metrics: self.metrics['loss'] = loss
        if 'acc' not in self.metrics: self.metrics['acc'] = loss
        if self.model_path and loss<self.metrics['loss'] and acc > self.metrics['acc']:
            print('save model: ',self.model_path)
            self.saver.save(sess, self.model_path)
           

    def train(self,sess=None):
        def work(sess):
            #try:
            #    self.saver.restore(sess, self.model_path)
            #    print('-------------------- model has already restored: ', self.model_path)
            #except:
            #    print('-------------------- model path no exists or model has changed, model path: ', self.model_path)
            #self.cv(sess)
            step=0
            for S,L,X,Y in self.cdp.batch_sample(batch_size=1000):
                fd={
                  self.we.inputs: np.array(X,dtype=np.int32),
                  self.brnn.sequence_length: np.array(L,dtype=np.int32),
                  self.clf.exp_label: np.array(Y,dtype=np.int32),
                }
                _,step,loss,acc = sess.run([self.train_op, self.clf.global_step, self.clf.loss, self.clf.acc],feed_dict=fd)
                print('step: %s, loss: %s, acc:%s'%(step,loss,acc))
                #每训练N次样本做一次交叉验证
                if step%100==0:
                    self.cv(sess)
        if sess==None:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                work(sess)
        else:
            work(sess)


if __name__=='__main__':
    cdp = CDP(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
      num_steps = 100,
    )
    m=BLSTMAttentionClassiffier(cdp=cdp, model_path='/root/tfNLP/motc/clf/dist_blstm_attention/model')
    m.train()
