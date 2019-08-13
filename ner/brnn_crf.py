import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from modeling.base_model import WordEmbedding, BiRnn, CRF
from data_processor.ner_processor import NerDataProcessor as NDP

class BrnnCrfNer():
    def __init__(self, num_step, num_words, num_tags, model_path=None):
        self.model_path=model_path
        self.num_step=num_step
        self.num_tags=num_tags
        self.num_words=num_words
        with tf.variable_scope('only_attention_classiffier'):
            self.we = WordEmbedding(num_step=num_step,dict_size=num_words,word_vec_size=50)
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
            self.crf = CRF(inputs=self.brnn.outputs,sequence_length=self.brnn.sequence_length)
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.crf.loss, global_step=self.crf.global_step)
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
          self.crf.exp_tags: np.array(Y,dtype=np.int32),
        }
        t0=time.time()
        step,loss,acc= sess.run([self.crf.global_step, self.crf.loss, self.crf.acc],feed_dict=fd)
        t1=time.time()
        print('----------------- global_step: ',step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        print('cross-validation ---------- loss: %s, acc:%s'%(loss,acc))
           

    def train(self,sess=None,train_generator=None,cv_generator=None,epochs=100):
        def work(sess):
            #self.cv(sess)
            for i in range(epochs):
                S,L,X,Y = next(train_generator)
                fd={
                  self.we.inputs: np.array(X,dtype=np.int32),
                  self.brnn.sequence_length: np.array(L),
                  self.crf.exp_tags: np.array(Y,dtype=np.int32),
                }
                _,step,loss,acc = sess.run([self.train_op, self.crf.global_step, self.crf.loss, self.crf.acc],feed_dict=fd)
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
    '''
    ndp = NDP(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
      num_step = 1000,
    )
    m=BrnnAttentionClassiffier(num_label=ndp.num_label,num_step=ndp.num_step,num_words=ndp.num_words, model_path='/root/tfNLP/motc/ner/brnn_crf/model')
    train_generator = ndp.batch_sample(batch_size=1000)
    cv_generator = ndp.batch_sample(batch_size=10000,work_type='cv')
    for i in range(100): m.train(train_generator=train_generator,cv_generator=cv_generator)
    '''
