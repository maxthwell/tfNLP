import tensorflow as tf
import numpy as np
import time
from tfNLP.classiffier.blstm_attention_classiffier import BLSTMAttentionClassiffier
from tfNLP.data_processor.processor import ClassiffierDataProcessor
from tfNLP.data_processor.processor import ClassiffierDataProcessor as CDP
 
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
 
# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
 
# Create and start a server for the local task.
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    cdp = CDP(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
      num_steps = 1000,
    )
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        #import pdb;pdb.set_trace()
        m=BLSTMAttentionClassiffier(cdp=cdp, model_path='/root/tfNLP/motc/clf/dist_blstm_attention/model')
        global_step = tf.Variable(0)
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

# Create a "supervisor", which oversees the training process.
sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                         logdir="/data/train_logs",
                         init_op=init_op,
                         summary_op=summary_op,
                         saver=m.saver,
                         global_step=global_step,#m.clf.global_step,
                         save_model_secs=600)
 
# The supervisor takes care of session initialization, restoring from
# a checkpoint, and closing when done or an error occurs.
with sv.managed_session(server.target) as sess:
    # Loop until the supervisor shuts down or 1000000 steps have completed.
    pass
    m.train(sess)
 
# Ask for all the services to stop.
sv.stop()
