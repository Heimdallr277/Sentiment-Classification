import datetime, os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model import NRRPUPM


#导入数据参数
tf.flags.DEFINE_integer("n_class", 5, "类别数量")
tf.flags.DEFINE_string("dataset", 'yelp13', "数据集")
tf.flags.DEFINE_integer('max_sen_len', 50, '每个句子包含最多的token数')
tf.flags.DEFINE_integer('max_doc_len', 40, '每个文档包含最多的token数')

tf.flags.DEFINE_integer("batch_size", 100, "每次训练选取的样本数")
tf.flags.DEFINE_string("checkpoint_dir", "", "训练时checkpoint目录")

#其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "如果指定的设备不存在，是否允许tf自动分配设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "是否打印设备分配日志")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#依次导入训练集、验证集、测试集
checkpoint_file = tf.train.latest_checkpoint("F://大三下//张伟-数据挖掘//pro_final//code//checkpoints//"+FLAGS.dataset+"//"+FLAGS.checkpoint_dir+"//")
testset = Dataset('data/'+FLAGS.dataset+'/test.ss')
# 不知道为什么使用相对路径一直报错，只好改成绝对路径了
with open("F://大三下//张伟-数据挖掘//pro_final//code//checkpoints//"+FLAGS.dataset+"//"+FLAGS.checkpoint_dir+"//wordsdict.txt", 'rb') as f:
    wordsdict = pickle.load(f)
with open("F://大三下//张伟-数据挖掘//pro_final//code//checkpoints//"+FLAGS.dataset+"//"+FLAGS.checkpoint_dir+"//usrdict.txt", 'rb') as f:
    usrdict = pickle.load(f)
with open("F://大三下//张伟-数据挖掘//pro_final//code//checkpoints//"+FLAGS.dataset+"//"+FLAGS.checkpoint_dir+"//prddict.txt", 'rb') as f:
    prddict = pickle.load(f)
print("{}.meta".format(checkpoint_file))
#测试集上创建batch
testset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)

graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        nrrpupm_userid = graph.get_operation_by_name("input/user_id").outputs[0]
        nrrpupm_productid = graph.get_operation_by_name("input/product_id").outputs[0]
        nrrpupm_input_x = graph.get_operation_by_name("input/input_x").outputs[0]
        nrrpupm_input_y = graph.get_operation_by_name("input/input_y").outputs[0]
        nrrpupm_sen_len = graph.get_operation_by_name("input/sen_len").outputs[0]
        nrrpupm_doc_len = graph.get_operation_by_name("input/doc_len").outputs[0]

        nrrpupm_accuracy = graph.get_operation_by_name("metrics/accuracy").outputs[0]
        nrrpupm_correct_num = graph.get_operation_by_name("metrics/correct_num").outputs[0]
        nrrpupm_mse = graph.get_operation_by_name("metrics/mse").outputs[0]

        def predict_step(u, p, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                nrrpupm_userid: u,
                nrrpupm_productid: p,
                nrrpupm_input_x: x,
                nrrpupm_input_y: y,
                nrrpupm_sen_len: sen_len,
                nrrpupm_doc_len: doc_len
            }
            accuracy, correct_num, mse = sess.run(
                [nrrpupm_accuracy, nrrpupm_correct_num, nrrpupm_mse],
                feed_dict)
            return correct_num, accuracy, mse

        #预测
        def predict(dataset, name=None):
            acc = 0
            rmse = 0.
            for i in range(dataset.epoch):
                correct_num, _, mse = predict_step(dataset.usr[i], dataset.prd[i], dataset.docs[i],
                                                   dataset.label[i], dataset.sen_len[i], dataset.doc_len[i], name)
                acc += correct_num
                rmse += mse
            acc = acc * 1.0 / dataset.data_size
            rmse = np.sqrt(rmse / dataset.data_size)
            return acc, rmse

        test_acc, test_rmse = predict(testset, name="test")
        print("\n测试准确度 %.4f    测试的RMSE: %.4f\n" % (test_acc, test_rmse))

