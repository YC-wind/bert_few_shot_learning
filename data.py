#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-12-27 20:45
"""
import re
import json
import random
import numpy as np
import tokenization
import warnings
from functools import wraps
from scipy.special import comb, perm
from sklearn.utils import shuffle

"""
事先确定好：
    每个类别的样本 要大于 K+Q，因为是无放回的采样

"""
tokenizer = tokenization.FullTokenizer(vocab_file="./bert_base/vocab.txt")


def deprecated(reason):
    def _decorator(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            warn_str = '%s is now deprecated! reason: %s' % (func.__name__, reason)
            warnings.warn(warn_str, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return __wrapper

    return _decorator


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_one_example(tokenizer, text_a, text_b=None, max_seq_len=256):
    l_a = len(text_a)
    l_b = len(text_b) if text_b else 0
    if l_a + l_b >= max_seq_len:
        cc = int((max_seq_len - l_b) / 2)
        text_a = text_a[:cc] + "。" + text_a[-cc:]
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[0:(max_seq_len - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


class FewShotClassificationData:
    def __init__(self, n, k, q, na_rate, sample_count=None):
        # n-way number of support class
        self.N = n
        # k-shot number of support sample per class
        self.K = k
        # query number of query sample per class
        self.Q = q
        # NA number of no class match
        self.na_rate = na_rate
        self.json_data = None
        self.tokens_data = None
        self.classes = None
        self.sample_count = sample_count

    @deprecated("please use get_batch_sample")
    def get_one_sample(self):
        """
        此处代码，参考 https://github.com/thunlp/FewRel 项目
        """
        target_classes = random.sample(self.classes, self.N)
        # N * K
        support_set = []
        # N * Q + na * Q
        query_set = []
        #
        query_label = []
        q_na = self.na_rate
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))
        # N
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.tokens_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                if count < self.K:
                    support_set.append(self.tokens_data[class_name][j])
                else:
                    query_set.append(self.tokens_data[class_name][j])
                count += 1
            # label 改成 multi label 类型
            label = np.zeros(self.N, np.int32)
            label[i] = 1
            query_label += [label] * self.Q

        # NA, 构造负样本，不在 support set 空间内
        for j in range(q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.tokens_data[cur_class]))),
                1, False)[0]
            query_set.append(self.tokens_data[cur_class][index])
        label = np.zeros(self.N, np.int32)
        # label[self.N] = 1
        query_label += [label] * q_na
        # 可以shuffle 一下query_set, query_label，不然都是顺序的看着别捏，有点慌(bert使用layer，影响应该不大)
        # 注意： support set 不能shuffle，需要保证好位置，因为后面要取 每个类 最相似的，或者平均的向量
        __ = []
        for q, l in zip(query_set, query_label):
            __.append((q, l))
        __ = shuffle(__)
        query_set = [_[0] for _ in __]
        query_label = [_[1] for _ in __]
        return support_set, query_set, query_label

    def get_batch_sample(self, batch_size=3):
        """
        此处代码，参考 https://github.com/thunlp/FewRel 项目
        """
        b_support_set, b_query_set, b_query_label = [], [], []
        for _ in range(batch_size):
            target_classes = random.sample(self.classes, self.N)
            # N * K
            support_set = []
            # N * Q + na * Q
            query_set = []
            #
            query_label = []
            q_na = self.na_rate
            na_classes = list(filter(lambda x: x not in target_classes,
                                     self.classes))
            # N
            for i, class_name in enumerate(target_classes):
                indices = np.random.choice(
                    list(range(len(self.tokens_data[class_name]))),
                    self.K + self.Q, False)
                count = 0
                for j in indices:
                    if count < self.K:
                        support_set.append(self.tokens_data[class_name][j])
                    else:
                        query_set.append(self.tokens_data[class_name][j])
                    count += 1
                # label 改成 multi label 类型
                label = np.zeros(self.N, np.int32)
                label[i] = 1
                query_label += [label] * self.Q

            # NA, 构造负样本，不在 support set 空间内
            for j in range(q_na):
                cur_class = np.random.choice(na_classes, 1, False)[0]
                index = np.random.choice(
                    list(range(len(self.tokens_data[cur_class]))),
                    1, False)[0]
                query_set.append(self.tokens_data[cur_class][index])
            label = np.zeros(self.N, np.int32)
            # label[self.N] = 1
            query_label += [label] * q_na
            # 可以shuffle 一下query_set, query_label，不然都是顺序的看着别捏，有点慌(bert使用layer，影响应该不大)
            # 注意： support set 不能shuffle，需要保证好位置，因为后面要取 每个类 最相似的，或者平均的向量
            __ = []
            for q, l in zip(query_set, query_label):
                __.append((q, l))
            __ = shuffle(__)
            query_set = [_[0] for _ in __]
            query_label = [_[1] for _ in __]
            b_support_set.extend(support_set)
            b_query_set.extend(query_set)
            b_query_label.extend(query_label)
        return b_support_set, b_query_set, b_query_label

    def prepare_data(self, path):
        """
        记录下 token，下次就不用每次都 tokenizer，也蛮耗时的，毕竟要token，查表
        后面只需读取，排列组合即可
        """
        # load data
        self.json_data = json.loads(open(path).read())
        self.tokens_data = {}
        for k, v in self.json_data.items():
            vv = []
            for i in v:
                feature = process_one_example(tokenizer, re.sub(r"\s+", "", i), None, max_seq_len=32)
                vv.append(feature)
            self.tokens_data[k] = vv

        self.classes = list(self.json_data.keys())
        # 当前采样的最佳数
        self.sample_count = self.sample_count if self.sample_count else comb(len(self.classes), self.N)
        print(self.sample_count)
        # self.get_one_sample()
        # self.get_batch_sample(3)


if __name__ == "__main__":
    fsc = FewShotClassificationData(5, 3, 2, 2)
    fsc.prepare_data("./data/train_data.json")
    # print(comb(50, 5))
