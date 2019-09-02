'''
TF_CONFIG='{"task":{"type":"ps", "index":0}, "cluster":{"ps":["10.2.174.11:2220"], "worker":["10.2.174.11:2221","10.2.174.11:2222"]}}' python3 tfjob.py
'''


import os,json
import tensorflow as tf
import numpy as np
import time
from data_processor.clf_processor import LocalFileClassiffierDataProcessor as CDP
from classiffier.only_attention_classiffier import OnlyAttentionClassiffier
from kubeflow.tfjob import DistributeMachine

class DistOnlyAttention(DistributeMachine):
    def __init__(self):
        super(DistOnlyAttention, self).__init__()

    def _create_tfmodel(self):
        self.dp = CDP(
            train_data_dir = os.getenv('TRAIN_DATA_PATH', '/data/THUCNews'),
            num_step = 1000,
        )
        return OnlyAttentionClassiffier(
            num_label=self.dp.num_label,
            num_step=self.dp.num_step,
            num_words=self.dp.num_words,
            model_path=os.getenv('MODEL_PATH', '/data/train_logs')
        )

    def _dist_train(self, sv):
        train_generator = self.dp.batch_sample(batch_size=100)
        cv_generator = self.dp.batch_sample(batch_size=1000, work_type='cv')
        step=0
        while step<100:
            step, loss, acc = self.tfmodel.train(generator = train_generator)
        if self.is_chief: 
            self.tfmodel.cv(generator = cv_generator)

def main(_):
    DistOnlyAttention()
 
if __name__=='__main__':
    tf.app.run(main=main)
