import os,sys,re,time,shutil
import asyncio
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict

class TFModel():
    def __init__(self, model_name, model_path):
        self.sess = None
        self.model_path = model_path
        self.response_queue = defaultdict(dict)
        self.request_queue  = defaultdict(dict)
        self.current_request_uid=0
        with tf.variable_scope(model_name):
            self.global_step = tf.Variable(0)
            self._build_model()
            self.saver = tf.train.Saver()

    def _build_model(self):
        pass

    def set_session(self, sess=None):
        if sess: self.sess = sess
        else:
            self.sess = tf.Session()

    def unset_session(self):
        self.sess = None

    def init_model(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
            
    def load_model(self):
        try:
            self.saver.restore(self.sess, self.model_path)
            print('-------------------- model has already restored: ', self.model_path)
            return True
        except:
            print('-------------------- model path no exists or model has changed, model path: ', self.model_path)
            return False

    def save_model(self):
        self.saver.save(self.sess, self.model_path)
        print('-------------------- model has already saved: ', self.model_path)
    
    #批量预测
    def batch_predict(self, feed_dict={}):
        raise Exception('you must realize function: batch_predict')

    #将预测输入转换成feed_dict
    def _to_feed_dict(self, inputs):
        raise Exception('you must realize function: _to_feed_dict')

    #异步预测
    async def apredict(self, **inputs):
          uid = self.current_request_uid
          self.current_request_uid += 1
          self.current_request_uid = self.current_request_uid%10000
          self.request_queue[uid] = self._to_feed_dict(inputs)
          try:
              self.response_queue.pop(uid)
          except:
              pass
          #10s内尝试100次获取预测结果
          for i in range(100):
              await asyncio.sleep(0.1)
              if uid in self.response_queue:
                  res = self.response_queue[uid]
                  self.response_queue.pop(uid)
                  #print(uid,res)
                  return res
          return None

    #主工作循环
    async def main_loop(self, batch_size=100):
        while True:
            await asyncio.sleep(0.1)
            uid_list=[]
            inputs_dict = defaultdict(list)
            for i in range(batch_size):
                try:
                    uid, inputs = self.request_queue.popitem()
                    uid_list.append(uid)
                    for name, data in inputs.items():
                        inputs_dict[name].append(data[np.newaxis,:])
                except:
                    break
            if len(uid_list) == 0:
                await asyncio.sleep(1)
                continue
            feed_dict = {}
            for name, data_list in inputs_dict.items():
                feed_dict[name] = np.concatenate(data_list)

            results = self.batch_predict(feed_dict=feed_dict)
            if not isinstance(results,dict):
                await asyncio.sleep(1)
                continue
            for name, outputs in results.items():
                for i in range(outputs.shape[0]):
                    uid = uid_list[i]
                    self.response_queue[uid][name]=outputs[i]
                    
