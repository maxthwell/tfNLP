import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from modeling.base_model import WordEmbedding, BiRnn, Classiffier, Attention
from data_processor.clf_processor import ClassiffierDataProcessor as CDP
from classiffier.evaluator import ClassiffierModelEvaluator

class BrnnAttentionClassiffier():
    def __init__(self, num_step, num_words, num_label, model_path=None):
        self.model_path=model_path
        self.num_step=num_step
        self.num_label=num_label
        self.num_words=num_words
        with tf.variable_scope('brnn_attention_classiffier'):
            self.we = WordEmbedding(num_step=num_step,dict_size=num_words,word_vec_size=50)
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
            attention = Attention(inputs=self.brnn.outputs)
            self.clf = Classiffier(inputs=attention.outputs, num_label=num_label)
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.clf.loss, global_step=self.clf.global_step)
            self.saver = tf.train.Saver()
            for var in tf.trainable_variables(): print(var.name)   

    def load_model(self,sess):
        try:
            self.saver.restore(sess, self.model_path)
            print('-------------------- model has already restored: ', self.model_path)
            return True
        except:
            print('-------------------- model path no exists or model has changed, model path: ', self.model_path)
            return False

    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self, sess,cv_generator,label_list=None):
        S,L,X,Y = next(cv_generator)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.brnn.sequence_length: np.array(L),
          self.clf.exp_label: np.array(Y,dtype=np.int32),
        }
        t0=time.time()
        step,loss,acc,all_label = sess.run([self.clf.global_step, self.clf.loss, self.clf.acc, self.clf.all_label],feed_dict=fd)
        t1=time.time()
        print('----------------- global_step: ',step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        eva = ClassiffierModelEvaluator(num_label=self.num_label,label_name_list=label_list)
        eva.batch_add(all_label)
        eva.update_quota()
        print('cross-validation ---------- loss: %s, acc:%s'%(loss,acc))
        print(eva)
           

    def train(self,sess=None,train_generator=None,cv_generator=None,epochs=100):
        def work(sess):
            #self.cv(sess)
            for i in range(epochs):
                S,L,X,Y = next(train_generator)
                fd={
                  self.we.inputs: np.array(X,dtype=np.int32),
                  self.brnn.sequence_length: np.array(L),
                  self.clf.exp_label: np.array(Y,dtype=np.int32),
                }
                _,step,loss,acc = sess.run([self.train_op, self.clf.global_step, self.clf.loss, self.clf.acc],feed_dict=fd)
                print('step: %s, loss: %s, acc:%s'%(step,loss,acc))
            print('save model: ',self.model_path)
            self.saver.save(sess, self.model_path)
            if cv_generator:
                self.cv(sess,cv_generator)
        if sess==None:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load_model(sess)
                work(sess)
        else:
            work(sess)


if __name__=='__main__':
    cdp = CDP(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
      num_step = 1000,
    )
    m=BrnnAttentionClassiffier(num_label=cdp.num_label,num_step=cdp.num_step,num_words=cdp.num_words, model_path='/root/tfNLP/motc/clf/brnn_attention/model')
    train_generator = cdp.batch_sample(batch_size=1000)
    cv_generator = cdp.batch_sample(batch_size=10000,work_type='cv')
    for i in range(100): m.train(train_generator=train_generator,cv_generator=cv_generator)
