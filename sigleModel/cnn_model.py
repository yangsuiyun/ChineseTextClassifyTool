# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        config = config

        # 三个待输入的数据
        input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [config.vocab_size, config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, config.num_filters, config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            logits = tf.layers.dense(fc, config.num_classes, name='fc2')
            y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
            loss = tf.reduce_mean(cross_entropy)
            # 优化器
            optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred_cls)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
