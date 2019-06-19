# ! /usr/bin/env python
# encoding: utf-8
import tensorflow as tf
import numpy as np
import os
from textcnn import TextCNN
import datetime
from utils import create_batch_data, create_word2vec, files_processing
def train(train_dir, val_dir, labels_file, word2vec_path, batch_size, max_steps, log_step, val_step, snapshot, out_dir):
    '''
    训练...
    :param train_dir: 训练数据目录
    :param val_dir:   val数据目录
    :param labels_file:  labels文件目录
    :param word2vec_path: 词向量模型文件
    :param batch_size: batch size
    :param max_steps:  最大迭代次数
    :param log_step:  log显示间隔
    :param val_step:  测试间隔
    :param snapshot:  保存模型间隔
    :param out_dir:   模型ckpt和summaries输出的目录
    :return:
    '''

    max_sentence_length = 300

    embedding_dim = 20

    filter_sizes = [3, 4, 5, 6]

    num_filters = 200  # Number of filters per filter size

    base_lr = 0.001  # 学习率

    dropout_keep_prob = 0.5

    l2_reg_lambda = 0.0  # "L2 regularization lambda (default: 0.0)

    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备

    log_device_placement = False  # 是否打印设备分配日志

    print("Loading data...")

    w2vModel = create_word2vec.load_wordVectors(word2vec_path)

    labels_set = files_processing.read_txt(labels_file)

    labels_nums = len(labels_set)

    train_file_list = create_batch_data.get_file_list(file_dir=train_dir, postfix='*.npy')

    train_batch = create_batch_data.get_data_batch(train_file_list, labels_nums=labels_nums, batch_size=batch_size,

                                                   shuffle=False, one_hot=True)

    val_file_list = create_batch_data.get_file_list(file_dir=val_dir, postfix='*.npy')

    val_batch = create_batch_data.get_data_batch(val_file_list, labels_nums=labels_nums, batch_size=batch_size,
                                                 shuffle=False, one_hot=True)

    print("train data info *****************************")

    train_nums = create_word2vec.info_npy(train_file_list)

    print("val data   info *****************************")

    val_nums = create_word2vec.info_npy(val_file_list)

    print("labels_set info *****************************")

    files_processing.info_labels_set(labels_set)

    # Training

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                      log_device_placement=log_device_placement)

        sess = tf.Session(config=session_conf)

        with sess.as_default():

            cnn = TextCNN(sequence_length=max_sentence_length,

                          num_classes=labels_nums,

                          embedding_size=embedding_dim,

                          filter_sizes=filter_sizes,

                          num_filters=num_filters,

                          l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure

            global_step = tf.Variable(0, name="global_step", trainable=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=base_lr)

            # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)

            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)

            grad_summaries = []

            for g, v in grads_and_vars:

                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)

                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

                    grad_summaries.append(grad_hist_summary)

                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries

            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy

            loss_summary = tf.summary.scalar("loss", cnn.loss)

            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries

            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

            train_summary_dir = os.path.join(out_dir, "summaries", "train")

            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):

                """

                A single training step

                """

                feed_dict = {

                    cnn.input_x: x_batch,

                    cnn.input_y: y_batch,

                    cnn.dropout_keep_prob: dropout_keep_prob

                }

                _, step, summaries, loss, accuracy = sess.run(

                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],

                    feed_dict)

                if step % log_step == 0:
                    print("training: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):

                """

                Evaluates model on a dev set

                """

                feed_dict = {

                    cnn.input_x: x_batch,

                    cnn.input_y: y_batch,

                    cnn.dropout_keep_prob: 1.0

                }

                step, summaries, loss, accuracy = sess.run(

                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],

                    feed_dict)

                if writer:
                    writer.add_summary(summaries, step)

                return loss, accuracy

            for i in range(max_steps):

                train_batch_data, train_batch_label = create_batch_data.get_next_batch(train_batch)

                train_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, train_batch_data)

                train_step(train_batch_data, train_batch_label)

                current_step = tf.train.global_step(sess, global_step)

                if i == max_steps-1:

                    val_losses = []

                    val_accs = []

                    # for k in range(int(val_nums/batch_size)):

                    for k in range(100):
                        val_batch_data, val_batch_label = create_batch_data.get_next_batch(val_batch)

                        val_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, val_batch_data)
                        print('val_batch_data是：',type(val_batch_data),val_batch_data.shape(),val_batch_data[0],len(val_batch_data))
                        val_loss, val_acc = dev_step(val_batch_data, val_batch_label, writer=dev_summary_writer)

                        val_losses.append(val_loss)

                        val_accs.append(val_acc)

                    mean_loss = np.array(val_losses, dtype=np.float32).mean()

                    mean_acc = np.array(val_accs, dtype=np.float32).mean()

                    print("--------Evaluation:step {}, loss {:g}, acc {:g}".format(current_step, mean_loss, mean_acc))

                if current_step % snapshot == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)

                    print("Saved model checkpoint to {}\n".format(path))


            def dev_step(x_batch, y_batch, writer=None):

                """

                Evaluates model on a dev set

                """

                feed_dict = {

                    cnn.input_x: x_batch,

                    cnn.input_y: y_batch,

                    cnn.dropout_keep_prob: 1.0

                }

                step, summaries, loss, accuracy = sess.run(

                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],

                    feed_dict)

                if writer:
                    writer.add_summary(summaries, step)

                return loss, accuracy


def main(dimen):
    # Data preprocess

    labels_file = '.\\data\\THUCNews_labels.txt'

    word2vec_path = "D:\\pypro\\bs\\model\\size50.model"

    max_steps = 10000  # 迭代次数

    batch_size = 128

    out_dir = ".\\models"  # 模型ckpt和summaries输出的目录

    train_dir = '.\\data\\train_data'

    val_dir = '.\\data\\val_data'
    '''
    train(train_dir=train_dir,

          val_dir=val_dir,

          labels_file=labels_file,

          word2vec_path=word2vec_path,

          batch_size=batch_size,

          max_steps=max_steps,

          log_step=50,

          val_step=500,

          snapshot=1000,

          out_dir=out_dir)
    '''
    model_dir = '.\\models\\checkpoints'
    saver = tf.train.import_meta_graph(model_dir+'\\model-40000.meta')
    graph = tf.get_default_graph()
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    x = graph.get_tensor_by_name('input_x:0')
    y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
    scores = graph.get_tensor_by_name('output/scores:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        print('加载成功！')
        labels_set = files_processing.read_txt(labels_file)
        labels_nums = len(labels_set)
        val_file_list = create_batch_data.get_file_list(file_dir=val_dir, postfix='*.npy')
        val_batch = create_batch_data.get_data_batch(val_file_list, labels_nums=labels_nums, batch_size=batch_size,
                                                     shuffle=False, one_hot=True)
        w2vModel = create_word2vec.load_wordVectors(word2vec_path)
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        val_losses = []
        val_accs = []
        for k in range(100):
            val_batch_data, val_batch_label = create_batch_data.get_next_batch(val_batch)

            val_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, val_batch_data)

            for i in val_batch_data:
                for j in i:
                    j[dimen]=0
                    '''
                    for d in range(dimen):
                        j[index[d]]=0
                    '''
            #print('val_batch_data是：', type(val_batch_data),val_batch_data[0].shape,  val_batch_data[0],len(val_batch_data[0]))
            feed_dict = {
                x: val_batch_data,
                y: val_batch_label,
                keep_prob: 1.0
            }
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y)
            loss = tf.reduce_mean(losses)
            val_loss = sess.run(loss, feed_dict)
            val_acc = sess.run(accuracy,feed_dict)
            #print('val_loss:',val_loss,'val_acc:',val_acc)

            val_losses.append(val_loss)

            val_accs.append(val_acc)

        mean_loss = np.array(val_losses, dtype=np.float32).mean()

        mean_acc = np.array(val_accs, dtype=np.float32).mean()
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(dimen,"--------Evaluation: loss {:g}, acc {:g}".format( mean_loss, mean_acc),nowTime)
        return mean_acc, mean_loss


if __name__ == "__main__":
    '''
    accfile = open('result40_single.txt','r',encoding='utf-8')
    lines = accfile.read().split('\n')
    acclist = []
    for i in lines:
        words = i.split(' ')
        acclist.append(float(words[0]))
    acclist = np.array(acclist)
    index = acclist.argsort()
    #index = index[::-1]
    '''
    for i in range(23,50):
        file = open('result50_single40000.txt', 'a', encoding='utf-8')
        #acc,loss = main(index,i)
        acc, loss = main(i)
        file.write('%f %f\n'%(acc,loss))
        file.close()
