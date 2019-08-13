import tensorflow as tf
import numpy as np
import time
from classiffier.blstm_attention_classiffier import BLSTMAttentionClassiffier
from data_processor.processor import ClassiffierDataProcessor
from data_processor.processor import ClassiffierDataProcessor as CDP

class DistributeMachine():
    def __init__(self): 
        # Flags for defining the tf.train.ClusterSpec
        tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
        tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
        # Flags for defining the tf.train.Server
        tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
        tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
        self.FLAGS = tf.app.flags.FLAGS

        self.ps_hosts = self.FLAGS.ps_hosts.split(",")
        self.worker_hosts = self.FLAGS.worker_hosts.split(",")
 
        # Create a cluster from the parameter server and worker hosts.
        self.cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
 
        # Create and start a server for the local task.
        self.server = tf.train.Server(self.cluster,
                         job_name=self.FLAGS.job_name,
                         task_index=self.FLAGS.task_index)

        if FLAGS.job_name == "ps":
            server.join()

    def build_model(self):
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.FLAGS.task_index,
                cluster=self.cluster)):
            self.custom_model()
            self.summary_op = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()

    def custom_model(self):
        cdp = CDP(
              train_data_dir='/data/THUCNews',
              test_data_dir='/data/THUCNewsTest',
              cv_data_dir='/data/THUCNewsTest',
              num_step = 1000,
        )
        self.model=BLSTMAttentionClassiffier(cdp=cdp, model_path='/root/tfNLP/motc/clf/dist_blstm_attention/model')

    def worker(self):
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                         logdir="/data/train_logs",
                         init_op=init_op,
                         summary_op=summary_op,
                         saver=self.model.saver,
                         global_step=self.model.global_step,
                         save_model_secs=600)
 
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            while not sv.should_stop():
                self.dist_work(sess)
 
        # Ask for all the services to stop.
        sv.stop()

    def dist_work(self,sess):
        self.model.
