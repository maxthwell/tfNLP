import os,sys,re,time,shutil
import random
import numpy as np
import tensorflow as tf
from tfnlp.modeling.base_model import WordEmbedding, BiRnn, Classiffier, Attention, Dropout
from tfnlp.data_processor.clf_processor import LocalFileClassiffierDataProcessor as CDP
from tfnlp.classiffier.evaluator import ClassiffierModelEvaluator
from tfnlp.modeling.tfmodel import TFModel
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__), path))

class OnlyAttentionClassiffier(TFModel):
    def __init__(self, num_step, num_words, num_label, model_name='only_attention_classiffier', model_path=None):
        self.model_path=model_path
        self.num_step=num_step
        self.num_label=num_label
        self.num_words=num_words
        super(OnlyAttentionClassiffier,self).__init__(model_name,model_path)
        
    def _build_model(self):
        self.we = WordEmbedding(num_step=self.num_step,dict_size=self.num_words,word_vec_size=50)
        attention = Attention(inputs=self.we.outputs)
        dropout = Dropout(inputs=attention.outputs, keep_prob=0.97)
        self.clf = Classiffier(inputs=dropout.outputs, num_label=self.num_label)
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.clf.loss, global_step=self.global_step)

    #做交叉验证，如果所有指标都比现有模型好则保持
    def cv(self, generator, label_list=None):
        S,L,X,Y = next(generator)
        fd={                                                                 
          self.we.inputs: np.array(X,dtype=np.int32),                        
          self.clf.exp_label: np.array(Y,dtype=np.int32),
        }
        t0=time.time()
        step,loss,acc,all_label = self.sess.run([self.global_step, self.clf.loss, self.clf.acc, self.clf.all_label],feed_dict=fd)
        t1=time.time()
        print('----------------- global_step: %s'%step)
        print('----------------- predict 10000 samples use time: %s sencond'%(t1-t0))
        eva = ClassiffierModelEvaluator(num_label=self.num_label,label_name_list=label_list)
        eva.batch_add(all_label)
        eva.update_quota()
        print('cross-validation ---------- loss: %s, acc:%s'%(loss,acc))
        print(eva)
           
    def export_model(self, export_dir):
        tf.saved_model.simple_save(self.sess, export_dir=export_dir, inputs={'wid':self.we.inputs}, outputs={'tags': self.clf.outputs})

    def train(self,generator=None):
        S,L,X,Y = next(generator)
        fd={
          self.we.inputs: np.array(X,dtype=np.int32),
          self.clf.exp_label: np.array(Y,dtype=np.int32),
        }
        _,step,loss,acc = self.sess.run([self.train_op, self.global_step, self.clf.loss, self.clf.acc],feed_dict=fd)
        print('step: %s, loss: %s, acc:%s'%(step,loss,acc))
        return step, loss, acc

    def set_dp(self,dp):
        self.dp = dp

    def batch_predict(self, feed_dict={}):
        outputs = self.sess.run(self.clf.outputs, feed_dict=feed_dict)
        return {'probe_dist': outputs}

    def _to_feed_dict(self, inputs):
        if 'content' in inputs:
            content = inputs['content']
            S,X = self.dp.process_text(content)
            return {self.we.inputs:np.array(X)}
        if 'word_list' in inputs:
            word_list = inputs['word_list']
            S,X = self.dp.process_word_list(word_list)
            return {self.we.inputs:np.array(X)}

if __name__=='__main__':
    dp = CDP(
      train_data_dir='/data/THUCNews',
      num_step = 1000,
    )
    m=OnlyAttentionClassiffier(num_label=dp.num_label,num_step=dp.num_step,num_words=dp.num_words, model_path=_get_module_path('../motc/clf/only_attention/model'))
    m.set_session()
    m.init_model()
    m.load_model()
    m.set_dp(dp)
    #import asyncio
    #tasks = [m.apredict(content='你好，我是你大爷') for i in range(100)] + [m.main_loop(batch_size=100)]
    #loop=asyncio.get_event_loop()
    #loop.run_until_complete(asyncio.wait(tasks))

    train_generator = dp.batch_sample(batch_size=100)
    cv_generator = dp.batch_sample(batch_size=1000,work_type='cv')
    for i in range(1000):
        for i in range(100):
            m.train(generator = train_generator)
        m.save_model()
        m.cv(generator = cv_generator)
