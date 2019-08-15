import os,sys,re,time,shutil
import random
import tensorflow as tf

class TFModel():
    def __init__(self, model_name, model_path):
        self.sess = None
        self.model_path = model_path
        with tf.variable_scope(model_name):
            self._build_model()
            self.saver = tf.train.Saver()

    def _build_model(self):
        pass

    def set_session(self, sess=None):
        if sess: self.sess = sess
        else:
            self.sess = tf.Session()

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

