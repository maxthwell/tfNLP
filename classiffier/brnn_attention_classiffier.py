import os,sys,re,time,shutil
import random
import numpy as np
import tensorflow as tf
from tfnlp.modeling.base_model import WordEmbedding, BiRnn, Classiffier, Attention
from tfnlp.data_processor.clf_processor import LocalFileClassiffierDataProcessor as CDP
from tfnlp.classiffier.evaluator import ClassiffierModelEvaluator
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__), path))

class BrnnAttentionClassiffier():
    def __init__(self, num_step, num_words, num_label, model_name='brnn_attention_classiffier', model_path=None):
        self.num_step=num_step
        self.num_label=num_label
        self.num_words=num_words
        super(BrnnAttentionClassiffier,self).__init__(model_name,model_path)

    def build_model(self):
        self.we = WordEmbedding(num_step=self.num_step,dict_size=self.num_words,word_vec_size=50)
        self.brnn = BiRnn(inputs=self.we.outputs, rnn_size_list=[30], rnn_type='gru')
        attention = Attention(inputs=self.brnn.outputs)
        self.clf = Classiffier(inputs=attention.outputs, num_label=self.num_label)
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.clf.loss, global_step=self.global_step)

    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self, generator, label_list=None):
        S,L,X,Y = next(generator)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.brnn.sequence_length: np.array(L),
          self.clf.exp_label: np.array(Y,dtype=np.int32),
        }
        t0=time.time()
        step,loss,acc,all_label = self.sess.run([self.global_step, self.clf.loss, self.clf.acc, self.clf.all_label],feed_dict=fd)
        t1=time.time()
        print('----------------- global_step: ',step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        eva = ClassiffierModelEvaluator(num_label=self.num_label,label_name_list=label_list)
        eva.batch_add(all_label)
        eva.update_quota()
        print('cross-validation ---------- loss: %s, acc:%s'%(loss,acc))
        print(eva)
           

    def train(self, generator=None, epochs=100):
        for i in range(epochs):
            S,L,X,Y = next(generator)
            fd={
              self.we.inputs: np.array(X,dtype=np.int32),
              self.brnn.sequence_length: np.array(L),
              self.clf.exp_label: np.array(Y,dtype=np.int32),
            }
            _,step,loss,acc = self.sess.run([self.train_op, self.global_step, self.clf.loss, self.clf.acc],feed_dict=fd)
            print('step: %s, loss: %s, acc:%s'%(step,loss,acc))
            self.save_model()

if __name__=='__main__':
    dp = CDP(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
      num_step = 1000,
    )
    m=BrnnAttentionClassiffier(num_label=dp.num_label,num_step=dp.num_step,num_words=dp.num_words, model_path=_get_module_path('../motc/clf/brnn_attention/model'))
    m.set_session()
    m.init_model()
    m.load_model()
    train_generator = dp.batch_sample(batch_size=1000)
    cv_generator = dp.batch_sample(batch_size=10000,work_type='cv')
    for i in range(100):
        m.train(generator=train_generator)
        m.cv(generator=cv_generator)
