'''
TF_CONFIG='{"task":{"type":"ps", "index":0}, "cluster":{"ps":["10.2.174.11:2220"], "worker":["10.2.174.11:2221","10.2.174.11:2222"]}}' python3 tfjob.py
'''


import os,json
import tensorflow as tf
import numpy as np
import time
from data_processor.clf_processor import LocalFileClassiffierDataProcessor as CDP
from classiffier.only_attention_classiffier import OnlyAttentionClassiffier

class DistributeMachine():
    def __init__(self):
        tf_config_json = os.environ.get("TF_CONFIG", "{}")
        tf_config = json.loads(tf_config_json)
        task = tf_config.get("task", {})
        self.cluster_spec = tf_config.get("cluster", {})
        self.job_name = task["type"]
        self.task_index = task["index"]

        # Create a cluster from the parameter server and worker hosts.
        self.cluster = tf.train.ClusterSpec(self.cluster_spec)
 
        # Create and start a server for the local task.
        self.server = tf.train.Server(
                         self.cluster,
                         job_name=self.job_name,
                         task_index=self.task_index
                      )

        if self.job_name == "ps":
            self.server.join()
        else:
            self.__build_model()
            self.worker()

    def __build_model(self):
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index,
                cluster=self.cluster)):
            self.tfmodel = self._create_tfmodel()
            self.summary_op = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()

    def _create_tfmodel(self):
        raise Exception("you must realize create_tfmodel function")

    def worker(self):
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                         logdir=self.tfmodel.model_path,
                         init_op=self.init_op,
                         summary_op=self.summary_op,
                         saver=self.tfmodel.saver,
                         global_step=self.tfmodel.global_step,
                         save_model_secs=60)
 
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(self.server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            self.tfmodel.set_session(sess)
            self._dist_train(sv)
            sv.stop()

    def _dist_train(self, sv):
        raise Exception("you must realize dist_train function")
 

if __name__=='__main__':
    machine = DistributeMachine()
