import os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model import NRRPUPM


#导入数据
tf.flags.DEFINE_integer("n_class", 5, "类别数量")
tf.flags.DEFINE_string("dataset", 'yelp13', "数据集")

#模型超参数
tf.flags.DEFINE_integer("embedding_dim", 200, "字符嵌入纬度")
tf.flags.DEFINE_integer("hidden_size", 100, "rnn隐藏状态维数")
tf.flags.DEFINE_integer('max_sen_len', 50, '每个句子包含最多的token数')
tf.flags.DEFINE_integer('max_doc_len', 40, '每个文档包含最多的token数')
tf.flags.DEFINE_float("lr", 0.005, "学习率")

#训练参数
tf.flags.DEFINE_integer("batch_size", 100, "每次训练选取的样本数")
tf.flags.DEFINE_integer("num_epochs", 1000, "循环整个数据集迭代次数")
tf.flags.DEFINE_integer("evaluate_every", 25, "经过多少步后，在验证集上评估模型")

#其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "如果指定的设备不存在，是否允许tf自动分配设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "是否打印设备分配日志")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


#依次导入训练集、验证集、测试集
print("Loading data...")
trainset = Dataset('../../data/'+FLAGS.dataset+'/train.ss')
devset = Dataset('../../data/'+FLAGS.dataset+'/dev.ss')
testset = Dataset('../../data/'+FLAGS.dataset+'/test.ss')

#导入词嵌入后的数据
alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)
embeddingpath = '../../data/'+FLAGS.dataset+'/embedding.txt'
embeddingfile, wordsdict = data_helpers.load_embedding(embeddingpath, alldata, FLAGS.embedding_dim)
del alldata
print("Loading data finished...")

usrdict, prddict = trainset.get_usr_prd_dict()#获取训练数据集的用户词典、产品词典
#迭代循环，一次训练取部分数据集
trainbatches = trainset.batch_iter(usrdict, prddict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
                                 FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)
devset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
testset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)


with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    #创建神经网络训练模型
    with sess.as_default():
        nrrpupm = NRRPUPM(
            max_sen_len = FLAGS.max_sen_len,
            max_doc_len = FLAGS.max_doc_len,
            class_num = FLAGS.n_class,
            embedding_file = embeddingfile,
            embedding_dim = FLAGS.embedding_dim,
            hidden_size = FLAGS.hidden_size,
            user_num = len(usrdict),
            product_num = len(prddict)
        )
        nrrpupm.build_model()
        #定义训练步骤
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)#使用adam优化
        grads_and_vars = optimizer.compute_gradients(nrrpupm.loss)#损失函数
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)#梯度优化

        #保存词典
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("../checkpoints/"+FLAGS.dataset+"/"+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with open(checkpoint_dir + "/wordsdict.txt", 'wb') as f:
            pickle.dump(wordsdict, f)#存储单词词典
        with open(checkpoint_dir + "/usrdict.txt", 'wb') as f:
            pickle.dump(usrdict, f)#存储用户词典
        with open(checkpoint_dir + "/prddict.txt", 'wb') as f:
            pickle.dump(prddict, f)#存储产品词典

        sess.run(tf.global_variables_initializer())

        #训练步骤
        def train_step(batch):
            u, p, x, y, sen_len, doc_len = zip(*batch)
            feed_dict = {
                nrrpupm.userid: u,
                nrrpupm.productid: p,
                nrrpupm.input_x: x,
                nrrpupm.input_y: y,
                nrrpupm.sen_len: sen_len,
                nrrpupm.doc_len: doc_len
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, nrrpupm.loss, nrrpupm.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        #预测步骤
        def predict_step(u, p, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                nrrpupm.userid: u,
                nrrpupm.productid: p,
                nrrpupm.input_x: x,
                nrrpupm.input_y: y,
                nrrpupm.sen_len: sen_len,
                nrrpupm.doc_len: doc_len
            }
            step, loss, accuracy, correct_num, mse = sess.run(
                [global_step, nrrpupm.loss, nrrpupm.accuracy, nrrpupm.correct_num, nrrpupm.mse],
                feed_dict)
            return correct_num, accuracy, mse

        #进行预测
        def predict(dataset, name=None):
            acc = 0
            rmse = 0.
            for i in xrange(dataset.epoch):
                correct_num, _, mse = predict_step(dataset.usr[i], dataset.prd[i], dataset.docs[i],
                                                   dataset.label[i], dataset.sen_len[i], dataset.doc_len[i], name)
                acc += correct_num
                rmse += mse
            acc = acc * 1.0 / dataset.data_size#计算准确率
            rmse = np.sqrt(rmse / dataset.data_size)#计算rmse用以模型评估
            return acc, rmse

        topacc = 0.
        toprmse = 0.
        better_dev_acc = 0.
        predict_round = 0

        #对每个batch循环进行训练
        for tr_batch in trainbatches:
            train_step(tr_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                predict_round += 1
                print("\n计算循环%d轮:" % (predict_round))

                dev_acc, dev_rmse = predict(devset, name="dev")
                print("验证集准确度: %.4f    验证集的RMSE: %.4f" % (dev_acc, dev_rmse))
                test_acc, test_rmse = predict(testset, name="test")
                print("测试集准确度: %.4f    测试集的RMSE: %.4f" % (test_acc, test_rmse))

                #输出验证集上最高的准确度
                if dev_acc >= better_dev_acc:
                    better_dev_acc = dev_acc
                    topacc = test_acc
                    toprmse = test_rmse
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("模型保存在{}\n".format(path))
                print("最高准确度: %.4f   最高RMSE: %.4f" % (topacc, toprmse))
