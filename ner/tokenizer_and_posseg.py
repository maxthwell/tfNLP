import os,sys,re,time,shutil
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from modeling.base_model import WordEmbedding, BiRnn, CRF
from modeling.tfmodel import TFModel
from data_processor.ner_processor import WikiDataProcessor as WDP
from data_processor.ner_processor import CorpusZhDataProcessor as CZDP

class TokenizerAndPosseg(TFModel):
    def __init__(self, num_step, num_words, model_name='TokenizerAndPosseg', model_path=None):
        self.model_path=model_path
        self.num_step=num_step
        self.num_words=num_words
        super(TokenizerAndPosseg,self).__init__(model_name,model_path) 

    def _build_model(self):
        self.we = WordEmbedding(num_step=self.num_step,dict_size=self.num_words,word_vec_size=50)
        self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
        self.crf_tokenizer = CRF(inputs=self.brnn.outputs,sequence_length=self.brnn.sequence_length,num_tags=2, name='crf_tokenizer')
        self.crf_posseg = CRF(inputs=self.brnn.outputs,sequence_length=self.brnn.sequence_length,num_tags=70, name='crf_posseg')
        self.train_op_tokenizer = tf.train.AdamOptimizer(1e-3).minimize(self.crf_tokenizer.loss, global_step=self.crf_tokenizer.global_step)
        self.train_op_posseg = tf.train.AdamOptimizer(1e-3).minimize(self.crf_posseg.loss, global_step=self.crf_posseg.global_step)
        for var in tf.trainable_variables(): print(var.name)

    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self, cv_generator,job='tokenizer'):
        if job == 'tokenizer': crf = self.crf_tokenizer
        elif job == 'posseg': crf = self.crf_posseg
        S,L,X,Y = next(cv_generator)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.brnn.sequence_length: np.array(L),
          crf.exp_tags: np.array(Y,dtype=np.int32),
        }
        step,loss,acc,tags= self.sess.run([crf.global_step, crf.loss, crf.acc, crf.outputs],feed_dict=fd)
        t0=time.time()
        self.sess.run([crf.global_step,crf.outputs],feed_dict=fd)
        t1=time.time()
        if job=='tokenizer':
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
           

    def train(self,train_generator=None,epochs=100, job='tokenizer'):
        if job=='tokenizer':
            crf=self.crf_tokenizer
            train_op=self.train_op_tokenizer
        elif job=='posseg':
            crf=self.crf_posseg
            train_op=self.train_op_posseg
            
        for i in range(epochs):
            S,L,X,Y = next(train_generator)
            fd={
                  self.we.inputs: np.array(X,dtype=np.int32),
                  self.brnn.sequence_length: np.array(L),
                  crf.exp_tags: np.array(Y,dtype=np.int32),
            }
            _,step,loss,acc,outputs = self.sess.run([train_op, crf.global_step, crf.loss, crf.acc, crf.outputs],feed_dict=fd)
            print('job: %s, step: %s, loss: %s, acc:%s'%(job,step,loss,acc))
        self.save_model()


if __name__=='__main__':
    tokenizer_dp = WDP(
      num_step = 1000,
      annotation_file='/data/wiki_corpus_chs.txt'
    )
    posseg_dp = CZDP(
      num_step = 1000,
      annotation_file='/data/corpusZh/B.txt'
    ) 
    m=TokenizerAndPosseg(num_step=tokenizer_dp.num_step,num_words=tokenizer_dp.num_words, model_path='/root/tfNLP/motc/ner/TokenizerAndPosseg/model')
    m.set_session()
    m.init_model()
    m.load_model()
    tokenizer_train_generator = tokenizer_dp.batch_sample(batch_size=1000)
    tokenizer_cv_generator = tokenizer_dp.batch_sample(batch_size=1000,work_type='cv')
    posseg_train_generator = posseg_dp.batch_sample(batch_size=1000)
    posseg_cv_generator = posseg_dp.batch_sample(batch_size=1000,work_type='cv')
    for i in range(10000):
        m.train(epochs=1,train_generator=posseg_train_generator, job='posseg')
        m.train(epochs=1,train_generator=tokenizer_train_generator, job='tokenizer')
        if i%20==0: 
            m.cv(cv_generator=tokenizer_cv_generator, job='tokenizer')    
            m.cv(cv_generator=posseg_cv_generator, job='posseg')    
            
