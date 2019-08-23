'''
TF_CONFIG='{"task":{"type":"ps", "index":0}, "cluster":{"ps":["10.2.174.11:2220"], "worker":["10.2.174.11:2221","10.2.174.11:2222"]}}' python3 tfjob.py
'''


import os,json
import tensorflow as tf
import numpy as np
import time
from data_processor.clf_processor import LocalFileClassiffierDataProcessor as CDP
from classiffier.only_attention_classiffier import OnlyAttentionClassiffier
from distributed.tfjob import DistributeMachine

class DistOnlyAttention(DistributeMachine):
    def __init__(self):
        super(DistOnlyAttention, self).__init__()

    def _create_tfmodel(self):
        self.dp = CDP(
            train_data_dir = '/data/THUCNews',
            num_step = 1000,
        )
        return OnlyAttentionClassiffier(
            num_label=self.dp.num_label,
            num_step=self.dp.num_step,
            num_words=self.dp.num_words,
            model_path='/data/train_logs'
        )

    def _dist_train(self, sv):
        train_generator = self.dp.batch_sample(batch_size=1000)
        cv_generator = self.dp.batch_sample(batch_size=10000,work_type='cv')
        while not sv.should_stop():
            step, loss, acc = self.tfmodel.train(generator = train_generator, epochs=50)
            if self.task_index == 0: 
                self.tfmodel.cv(generator = cv_generator)
            if step>100:
                self.tfmodel.unset_session()
                break
 
if __name__=='__main__':
    machine = DistOnlyAttention()
