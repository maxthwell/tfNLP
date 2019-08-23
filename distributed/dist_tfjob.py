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
            self.create_tfmodel()
            self.summary_op = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()

    def create_tfmodel(self):
        self.dp = CDP(
            train_data_dir = '/data/THUCNews',
            num_step = 1000,
        )
        self.tfmodel=OnlyAttentionClassiffier(
            num_label=self.dp.num_label,
            num_step=self.dp.num_step,
            num_words=self.dp.num_words,
            model_path='/root/tfNLP/motc/clf/only_attention/model'
        )

    def worker(self):
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                         logdir="/data/train_logs",
                         init_op=self.init_op,
                         summary_op=self.summary_op,
                         saver=self.tfmodel.saver,
                         global_step=self.tfmodel.global_step,
                         save_model_secs=60)
 
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(self.server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            train_generator = self.dp.batch_sample(batch_size=1000)
            cv_generator = self.dp.batch_sample(batch_size=10000,work_type='cv')
            self.tfmodel.set_session(sess)
            while not sv.should_stop():
                step, loss, acc = self.tfmodel.train(generator = train_generator, epochs=5)
                if self.task_index == 0: 
                    self.tfmodel.cv(generator = cv_generator)
                if step>100:
                    sv.request_stop()
                    self.tfmodel.unset_session()
 
        # Ask for all the services to stop.
        print("--------------sv.stop()")
        sv.stop()

if __name__=='__main__':
    machine = DistributeMachine()
