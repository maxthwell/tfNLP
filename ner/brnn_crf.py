import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from modeling.base_model import WordEmbedding, BiRnn, CRF
from data_processor.ner_processor import WikiDataProcessor as WDP

class BrnnCrfNer():
    def __init__(self, num_step, num_words, num_tags, model_path=None):
        self.model_path=model_path
        self.num_step=num_step
        self.num_tags=num_tags
        self.num_words=num_words
        self.sess=None
        with tf.variable_scope('brnn_crf_ner'):
            self.we = WordEmbedding(num_step=num_step,dict_size=num_words,word_vec_size=50)
            self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
            self.crf = CRF(inputs=self.brnn.outputs,sequence_length=self.brnn.sequence_length)
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.crf.loss, global_step=self.crf.global_step)
            self.saver = tf.train.Saver()
            for var in tf.trainable_variables(): print(var.name)   

    def set_session(self, sess=None):
        if sess: self.sess = sess
        else:
            self.sess = tf.Session()

    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
            
    def load_model(self):
        try:
            self.saver.restore(self.sess, self.model_path)
            print('-------------------- model has already restored: ', self.model_path)
            return True
        except:
            print('-------------------- model path no exists or model has changed, model path: ', self.model_path)
            return False

    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self,cv_generator,label_list=None):
        S,L,X,Y = next(cv_generator)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.brnn.sequence_length: np.array(L),
          self.crf.exp_tags: np.array(Y,dtype=np.int32),
        }
        step,loss,acc,tags= self.sess.run([self.crf.global_step, self.crf.loss, self.crf.acc,self.crf.outputs],feed_dict=fd)
        t0=time.time()
        self.sess.run([self.crf.global_step,self.crf.outputs],feed_dict=fd)
        t1=time.time()
        for batch_id in range(100):
            tag = tags[batch_id]
            sentence = S[batch_id]
            s_list=[]
            for id_s in range(len(sentence)):
                if tag[id_s]==1: s_list.append('_')
                s_list.append(sentence[id_s])
            print(''.join(s_list))
        print('----------------- deal string length: ', sum(L))        
        print('----------------- global_step: ',step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        print('----------------- cross-validation loss: %s, acc:%s'%(loss,acc))
           

    def train(self,train_generator=None,cv_generator=None,epochs=100):
        for i in range(epochs):
            S,L,X,Y = next(train_generator)
            fd={
                  self.we.inputs: np.array(X,dtype=np.int32),
                  self.brnn.sequence_length: np.array(L),
                  self.crf.exp_tags: np.array(Y,dtype=np.int32),
            }
            _,step,loss,acc,outputs = self.sess.run([self.train_op, self.crf.global_step, self.crf.loss, self.crf.acc,self.crf.outputs],feed_dict=fd)
            print('step: %s, loss: %s, acc:%s'%(step,loss,acc))
        print('save model: ',self.model_path)
        self.saver.save(self.sess, self.model_path)
        if cv_generator:
            self.cv(self.sess,cv_generator)


if __name__=='__main__':
    wdp = WDP(
      num_step = 1000,
      annotation_file='/data/wiki_corpus_chs.txt'
    )
    m=BrnnCrfNer(num_tags=2,num_step=wdp.num_step,num_words=wdp.num_words, model_path='/root/tfNLP/motc/ner/brnn_crf/model')
    m.set_session()
    m.init_model()
    train_generator = wdp.batch_sample(batch_size=1000)
    cv_generator = wdp.batch_sample(batch_size=1000,work_type='cv')
    for i in range(10000): m.train(epochs=10,train_generator=train_generator,cv_generator=cv_generator)
    
            
            
