#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-10-17 16:55
"""
import os, math, json
import tensorflow as tf
import numpy as np
import modeling
import optimization  # _freeze as optimization
from data import FewShotClassificationData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = {
    "in_1": "./data/train_data.json",  # 第一个输入为 训练文件
    "in_2": "./data/dev_data.json",  # 第二个输入为 验证文件
    "bert_config": "./bert_base/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./bert_base/bert_model.ckpt",  # 预训练bert模型
    "train_iter": 50000,
    "dev_iter": 100,
    "batch_size": 3,
    "N": 5,  # n-way support set下，类别的个数
    "K": 3,  # k-shot support set下，每个类下面query的个数
    "Q": 2,  # q 每个类下面query的个数
    "O": 2,  # other 不属于 support set 里面的个数
    "eval_start_step": 1000,
    "eval_per_step": 500,
    "auto_save": 1000,
    "margin": 10,
    "learning_rate": 1e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 32,  # 输入文本片段的最大 char级别 长度
    "out": "./fsl_proto_euclid/",  # 保存模型路径
    "out_1": "./fsl_proto_euclid/"  # 保存模型路径
}


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def contrastive_loss(y, d, M):
    """
    (B * total_Q * N)
    避免梯度爆炸，有可能存在 距离很大的样本的，这里限制下，已经很大的loss 了
    """
    # tmp = y * tf.square(d)
    tmp = tf.multiply(y, tf.square(tf.minimum(d, config["margin"])))
    # tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    tmp2 = tf.multiply((1 - y), tf.square(tf.maximum((config["margin"] - d), 0)))
    # sum
    sum = tf.reduce_sum(tf.add(tmp, tmp2))
    print(tmp, tmp2, sum)
    return sum / config["batch_size"] / M / 2


def get_euclidean_dist(x, y):
    """
    输入：(k, n)，(m, n)
    输出：(k, m)
    """
    square_x = tf.reduce_sum(tf.square(x), axis=-1)
    square_y = tf.reduce_sum(tf.square(y), axis=-1)
    # expand dims for broadcasting
    ex = tf.expand_dims(square_x, axis=-1)
    ey = tf.expand_dims(square_y, axis=-2)
    # XY matrix
    # xy = tf.einsum('bij,bkj->bik', x, y)
    # 如果没有batch_size这个维度，可以写成：
    xy = tf.einsum('ij,kj->ik', x, y)
    # compute distance，浮点防溢出
    dist = tf.sqrt(ex - 2 * xy + ey + 1e-10)
    return dist


def get_euclidean_dist_batch(x, y):
    """
    输入：(b, k, n)，(b, m, n)
    输出：(b, k, m)
    """
    square_x = tf.reduce_sum(tf.square(x), axis=-1)
    square_y = tf.reduce_sum(tf.square(y), axis=-1)
    # expand dims for broadcasting
    ex = tf.expand_dims(square_x, axis=-1)
    ey = tf.expand_dims(square_y, axis=-2)
    # XY matrix
    xy = tf.einsum('bij,bkj->bik', x, y)
    # 如果没有batch_size这个维度，可以写成：
    # xy = tf.einsum('ij,kj->ik', x, y)
    # compute distance，浮点防溢出
    dist = tf.sqrt(ex - 2 * xy + ey + 1e-10)
    return dist


def get_cos_distance(x1, x2):
    """
    输入：(k, n)，(m, n)
    输出：(k, m)
    """
    # calculate cos distance between two sets
    # more similar more big
    (k, n) = x1.shape
    (m, n) = x2.shape
    # 求模
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    # 内积
    x1_x2 = tf.matmul(x1, tf.transpose(x2))
    x1_x2_norm = tf.matmul(tf.reshape(x1_norm, [k, 1]), tf.reshape(x2_norm, [1, m]))
    # 计算余弦距离
    cos = x1_x2 / x1_x2_norm
    return cos


def get_cos_distance_batch(x1, x2):
    """
    输入：(b, k, n)，(b, m, n)
    输出：(b, k, m)
    """
    # calculate cos distance between two sets
    # more similar more big
    (b_, k, n) = x1.shape
    (b_, m, n) = x2.shape
    print(b_, m, n, k)
    # 求模
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1))
    print(x1_norm, x2_norm)
    # 内积
    x1_x2 = tf.matmul(x1, tf.transpose(x2, perm=[0, 2, 1]))
    x1_x2_norm = tf.matmul(tf.expand_dims(x1_norm, 2), tf.expand_dims(x2_norm, 1))
    print(x1_x2, x1_x2_norm)
    # 计算余弦距离
    cos = tf.div(x1_x2, x1_x2_norm)
    return cos


def create_model(bert_config, is_training, input_ids_support,
                 input_mask_support, segment_ids_support,
                 input_ids_query, input_mask_query,
                 segment_ids_query, labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model_support = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_support,
        input_mask=input_mask_support,
        token_type_ids=segment_ids_support,
        use_one_hot_embeddings=use_one_hot_embeddings)

    model_query = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_query,
        input_mask=input_mask_query,
        token_type_ids=segment_ids_query,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model_support.get_pooled_output()
    output_layer = modeling.layer_norm(output_layer, name="layer_norm")
    print("output_layer(support):{}".format(output_layer.shape))

    output_layer2 = model_query.get_pooled_output()
    output_layer2 = modeling.layer_norm(output_layer2, name="layer_norm2")
    hidden_size = output_layer2.shape[-1]
    print("output_layer(query):{}, hidden_size:{}".format(output_layer2.shape, hidden_size))
    # if is_training:
    #     # I.e., 0.1 dropout
    #     output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)
    #     output_layer2 = tf.nn.dropout(output_layer2, keep_prob=keep_prob)
    total_q = config["N"] * config["Q"] + config["O"]
    support = tf.reshape(output_layer, [-1, config["N"], config["K"], hidden_size])  # (B, N * K, D)
    batch_size = support.shape[0]
    # 取均值作为 proto embedding
    support = tf.reduce_mean(support, axis=2)  # (B, N  D)
    query = tf.reshape(output_layer2, [-1, total_q, hidden_size])  # (B, total_Q, D)

    print("use euclidean_dist...")
    print("support:{}, query:{}".format(support, query))
    euclidean_distance = get_euclidean_dist_batch(query, support)  # (B, total_Q, N)
    euclidean_distance = tf.reshape(euclidean_distance, [-1, config["N"]])
    print("euclidean_distance:{}".format(euclidean_distance))
    logits = euclidean_distance
    # 距离折中， 也取1，原始取的是 soft max， 这里为了适应多标签，改造了下
    pred = tf.cast(tf.less_equal(logits, tf.constant(float(config["margin"]), dtype=tf.float32)), tf.int32,
                   name="predictions")
    print("pred:{}, logits:{},".format(pred, logits))
    loss = contrastive_loss(tf.reshape(tf.cast(labels, dtype=tf.float32), [-1]),
                            tf.reshape(euclidean_distance, [-1]),
                            total_q)
    return logits, pred, loss


def main():
    print("print start load the params...")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(config["out"])
    learning_rate = config["learning_rate"]
    num_warmup_steps = math.ceil(config["train_iter"] * config["warmup_proportion"])

    use_one_hot_embeddings = False
    is_training = True
    use_tpu = False
    seq_len = config["max_seq_len"]
    init_checkpoint = config["init_checkpoint"]
    print("print start compile the bert model...")
    # 定义输入输出
    # support set (B * N * K, seq), where seq is seq_len
    input_ids_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_support')
    input_mask_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_support')
    segment_ids_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_support')
    # query set  (B * total_Q, seq)
    input_ids_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_query')
    input_mask_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_query')
    segment_ids_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_query')
    # label set (B * total_Q)
    labels = tf.placeholder(tf.int64, shape=[None, config["N"]], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])
    (logits, pred, loss) = create_model(bert_config_, is_training, input_ids_support,
                                        input_mask_support, segment_ids_support,
                                        input_ids_query, input_mask_query,
                                        segment_ids_query, labels, False)

    exit(0)
    fsc_train = FewShotClassificationData(config["N"], config["K"], config["Q"], config["O"], )
    fsc_train.prepare_data(config["in_1"])
    fsc_dev = FewShotClassificationData(config["N"], config["K"], config["Q"], config["O"], )
    fsc_dev.prepare_data(config["in_2"])

    train_op = optimization.create_optimizer(
        loss, learning_rate, config["train_iter"], num_warmup_steps, False)
    print("print start train the bert model(few shot learning)...")

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
                           max_to_keep=2)  # 保存最后top3模型

    with tf.Session() as sess:
        sess.run(init_global)
        print("start load the pre train model")

        if init_checkpoint:
            # tvars = tf.global_variables()
            tvars = tf.trainable_variables()
            print("global_variables", len(tvars))
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            print("initialized_variable_names:", len(initialized_variable_names))
            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, init_checkpoint)
            tvars = tf.global_variables()
            initialized_vars = [v for v in tvars if v.name in initialized_variable_names]
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            tf.logging.info('--all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            for v in initialized_vars:
                print('--initialized: %s, shape = %s' % (v.name, v.shape))
            for v in not_initialized_vars:
                print('--not initialized: %s, shape = %s' % (v.name, v.shape))
        else:
            sess.run(tf.global_variables_initializer())
        # if init_checkpoint:
        #     saver.restore(sess, init_checkpoint)
        #     print("checkpoint restored from %s" % init_checkpoint)
        print("********* bert_multi_class_train start *********")

        # tf.summary.FileWriter("output/",sess.graph)
        # albert remove dropout
        def train_step(ids_support, mask_support, segment_support, ids_query, mask_query, segment_query, y, step):
            """
            acc 是要全部一致，才会 记对
            """
            feed = {input_ids_support: ids_support,
                    input_mask_support: mask_support,
                    segment_ids_support: segment_support,
                    input_ids_query: ids_query,
                    input_mask_query: mask_query,
                    segment_ids_query: segment_query,
                    labels: y,
                    keep_prob: 1.0}
            _, l, pred_, loss_ = sess.run([train_op, logits, pred, loss], feed_dict=feed)
            count = np.sum((np.sum(np.equal(pred_, y), -1) >= config["N"]).astype(np.int16))
            acc_ = float(count) / len(y)
            print("step :{}, lr:{}, loss :{}, acc :{}".format(step, _[1], loss_, acc_))
            return loss_, pred_

        def dev_step(ids_support, mask_support, segment_support, ids_query, mask_query, segment_query, y):
            feed = {input_ids_support: ids_support,
                    input_mask_support: mask_support,
                    segment_ids_support: segment_support,
                    input_ids_query: ids_query,
                    input_mask_query: mask_query,
                    segment_ids_query: segment_query,
                    labels: y,
                    keep_prob: 0.9
                    }
            l, pred_, loss_ = sess.run([logits, pred, loss], feed_dict=feed)
            count = np.sum((np.sum(np.equal(pred_, y), -1) >= config["N"]).astype(np.int16))
            acc_ = float(count) / len(y)
            print("loss :{}, acc :{}".format(loss_, acc_))
            return loss_, pred_

        min_total_loss_dev = 999999
        step = 0
        # 动态生成数据，采样 support set 及 query set
        for epoch in range(1):
            _ = "{:*^100s}".format(("epoch-" + str(epoch).center(20)))
            print(_)
            # 读取训练数据
            for i in range(config["train_iter"]):
                step += 1
                # 训练步骤
                support_set, query_set, query_label = fsc_train.get_batch_sample(config["batch_size"])
                ids_support_ = [_[0] for _ in support_set]
                mask_support_ = [_[1] for _ in support_set]
                segment_support_ = [_[2] for _ in support_set]

                ids_query_ = [_[0] for _ in query_set]
                mask_query_ = [_[1] for _ in query_set]
                segment_query_ = [_[2] for _ in query_set]

                train_step(ids_support_, mask_support_, segment_support_, ids_query_, mask_query_, segment_query_,
                           query_label, step)

                if step % config["eval_per_step"] == 0 and step >= config["eval_start_step"]:
                    total_loss_dev = 0
                    # 验证步骤
                    total_pre_dev = []
                    total_true_dev = []
                    for j in range(config["dev_iter"]):  # 一个 epoch 的 轮数
                        # 验证操作
                        support_set, query_set, query_label = fsc_dev.get_batch_sample(config["batch_size"])

                        ids_support_ = [_[0] for _ in support_set]
                        mask_support_ = [_[1] for _ in support_set]
                        segment_support_ = [_[2] for _ in support_set]

                        ids_query_ = [_[0] for _ in query_set]
                        mask_query_ = [_[1] for _ in query_set]
                        segment_query_ = [_[2] for _ in query_set]

                        out_loss, pre = dev_step(ids_support_, mask_support_, segment_support_, ids_query_, mask_query_,
                                                 segment_query_, query_label)

                        total_loss_dev += out_loss
                        total_pre_dev.extend(pre)
                        total_true_dev.extend(query_label)
                    #
                    print("dev result report:")
                    # print(classification_report(total_true_dev, total_pre_dev, digits=4))

                    if total_loss_dev < min_total_loss_dev:
                        print("save model:\t%f\t>%f" % (min_total_loss_dev, total_loss_dev))
                        min_total_loss_dev = total_loss_dev
                        saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
                elif step < config["eval_start_step"] and step % config["auto_save"] == 0:
                    saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
            _ = "{:*^100s}".format("epoch-" + str(epoch) + "report:".center(20))
            print(_)
            # print("total_loss_train:{}".format(total_loss_train))
            # print(classification_report(total_true_train, total_pre_train, digits=4))
    sess.close()

    # remove dropout

    print("remove dropout in predict")
    tf.reset_default_graph()
    is_training = False
    input_ids_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_support')
    input_mask_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_support')
    segment_ids_support = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_support')
    # query set  (B * total_Q, seq)
    input_ids_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_query')
    input_mask_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_query')
    segment_ids_query = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_query')
    # label set (B * total_Q)
    labels = tf.placeholder(tf.int64, shape=[None, config["N"]], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])
    (logits, pred, loss) = create_model(bert_config_, is_training, input_ids_support,
                                        input_mask_support, segment_ids_support,
                                        input_ids_query, input_mask_query,
                                        segment_ids_query, labels, False)

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存最后top3模型

    try:
        checkpoint = tf.train.get_checkpoint_state(config["out"])
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = config["out"]
        print("[INFO] Model folder", config["out"], repr(e))

    with tf.Session() as sess:
        sess.run(init_global)
        saver.restore(sess, input_checkpoint)
        saver.save(sess, config["out_1"] + 'bert.ckpt')
    sess.close()


if __name__ == "__main__":
    print("********* fsl proto model start *********")
    main()
