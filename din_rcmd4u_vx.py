#!/usr/bin/env python
# coding=utf-8

import glob
import os
import random
import shutil
import sys

import tensorflow as tf

# 获取到trainer路径并加入sys.path
# __file__返回py文件绝对路径C:/trainer/estimator/din_rcmd4u_v1.py,再用os.path.dirname(__file__)返回文件父目录C:/trainer/estimator
trainer_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(trainer_path)

from neuralnet.input_layer import LRLayer, ConcatLayer, EmbeddingLayer
from neuralnet.feature_parse import build_feature_spec, TFRecordParser, load_sample_spec
from neuralnet import embedding, attention

root = 'C:/Users/N14369/Desktop/model_test/rank/din_recom4u_v4'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("task", 'train', "两种任务选择:train,export")
tf.app.flags.DEFINE_string("feature_spec", os.path.join(root, 'configure/tensor_spec.json'), "feature & label的数据元信息")
tf.app.flags.DEFINE_string("data", os.path.join(root, 'tfrecords'), "train & validate数据集所在目录")
tf.app.flags.DEFINE_string("checkpoint", os.path.join(root, 'checkpoint'), "checkpoint保存目录")
tf.app.flags.DEFINE_string("saved_model", os.path.join(root, 'saved_model'), "pb模型文件导出目录")

tf.app.flags.DEFINE_integer("epoch", 3, "训练集遍历次数")
tf.app.flags.DEFINE_integer("batch", 512, "batch大小")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "学习率")
tf.app.flags.DEFINE_string("gate_units", '16', "gate_layers structure")
tf.app.flags.DEFINE_string("expert_units", '128,64,32', "expert_layers structure")
tf.app.flags.DEFINE_string("tower_units", '8', "deep layer structure")
tf.app.flags.DEFINE_integer("dcn_num", '1', "expert_num")
tf.app.flags.DEFINE_integer("gate_num", '2', "gate_num")
tf.app.flags.DEFINE_integer("task_num", '4', "task_num")
tf.app.flags.DEFINE_integer("rng", None, "随机数种子,用于重复实验")

#定义交叉网络
def cross_layer(input_layer, x, name, idx_name):
    with tf.variable_scope(idx_name):
        with tf.variable_scope(name):
            input_dim = input_layer.get_shape().as_list()[1]
            w = tf.get_variable("cross_weight", [input_dim, input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.keras.regularizers.l2(0.000001))
            b = tf.get_variable("cross_bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=tf.keras.regularizers.l2(0.000001))
            xb = tf.matmul(w, tf.transpose(x))
            return input_layer * tf.transpose(xb) + b + x
#建立交叉网络
def build_cross_layers(input_layer, num_crosslayers, idx):
    x = input_layer
    for i in range(num_crosslayers):
        x = cross_layer(input_layer, x, 'cross_{}'.format(i), 'dcn_{}'.format(idx))
    return x

#定义专家网络
def Expert(input_layer, expert_units, mode):
    net = input_layer
    for units in expert_units:
        net = tf.layers.dense(net, units=units, activation=None, use_bias=False,
                                kernel_initializer=tf.glorot_normal_initializer(),
                                #kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)
        net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
    return net

#定义专家组网络
def ExpertGroup(input_layer, expert_units, expert_num, mode):
    experts = [expert_units]*expert_num
    expert_out, mmoe_out = None, []
    for expert in experts:
        net = input_layer
        for units in expert:
            net = tf.layers.dense(inputs=net, units=units, activation=None, use_bias=False,
                                          kernel_initializer=tf.glorot_normal_initializer(),
             #                             kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                          trainable=mode == tf.estimator.ModeKeys.TRAIN)
            net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
            net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, 1, expert[-1]])

        # 专家网络输出: -1, expert_num, expert_layer[-1]
        if expert_out is None:
            expert_out = net
        else:
            expert_out = tf.concat([expert_out, net], axis=-1)
    return expert_out

#定义Gate网络
def Gate(input_layer, gate_units, n_experts, mode):
    net = input_layer
    for units in gate_units:
        net = tf.layers.dense(net, units=units, activation=None, use_bias=False,
                                kernel_initializer=tf.glorot_normal_initializer(),
              #                  kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)
        net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
    logits = tf.layers.dense(net, n_experts, activation=None)
    probabilities = tf.nn.softmax(logits)
    return probabilities

#定义tower网络
def Tower(input_layer, tower_units, mode):
    net = input_layer
    for units in tower_units:
        net = tf.layers.dense(net, units=units, activation=None, use_bias=False,
                                kernel_initializer=tf.glorot_normal_initializer(),
        #                        kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)
        net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
    return net

def logloss(weight, labels, predictions):
    logloss = (weight**labels) * (-(labels * tf.log(predictions + 1e-7) + (1 - labels) * tf.log(1 - predictions + 1e-7)))
    return logloss

#Define focal loss
def focal_loss(predictions, labels, weight, alpha, gamma):
    zeros = tf.zeros_like(predictions, dtype=predictions.dtype)
    pos_corr = tf.where(labels > zeros, labels - predictions, zeros)
    neg_corr = tf.where(labels > zeros, zeros, predictions)
    fl_loss =  (weight**labels) * (-alpha * (pos_corr**gamma)*tf.log(predictions) - (1-alpha)*(neg_corr**gamma)*tf.log(1.0 - predictions))
    #return tf.divide(tf.reduce_sum(fl_loss), 512)
    #return tf.reduce_mean(fl_loss)
    #return tf.reduce_sum(fl_loss)
    return fl_loss


def model_fn(features, labels, mode, params):
    """Build Estimator Model function"""
    feature_conf = params['feature_conf']
    label_conf = params['label_conf']
    expert_units = params['expert_units']
    gate_units = params['gate_units']
    tower_units = params['tower_units']
    dcn_num = params['dcn_num']
    gate_num = params['gate_num']
    task_num = params['task_num']

    # 1.创建LR模型
    lr_layer = LRLayer(scope_name='lr_input', l1_reg=tf.contrib.layers.l1_regularizer(0.00001))
    lr_layer.add_sparse('lr_x_bool_fts', features.get('x_bool_fts'), feature_conf.get('x_bool_fts').size)
    lr_layer.add_sparse('lr_x_gt', features.get('x_gt'), feature_conf.get('x_gt').size)
    lr_layer.add_sparse('lr_a_hot', features.get('a_hot'), feature_conf.get('a_hot').size)

    lr_position = LRLayer(scope_name='lr_position', l1_reg=tf.contrib.layers.l1_regularizer(0.00001))
    lr_position.add_sparse('lr_position', features.get('position_id'), feature_conf.get('position_id').size,tf.cast(tf.cast(features.get('position_id'), dtype=tf.bool), dtype=tf.float32))
    lr_position_out = lr_position.forward()

    lr_out = lr_layer.forward()

    # 2.创建Deep模型(embedding.shape=[-1,length,dimension])
    # 2.1 Embedding Layer把embedding都初始化好
    embedding_layer = EmbeddingLayer(scope_name='embedding', l2_reg=tf.contrib.layers.l2_regularizer(0.000001))
    embedding_layer.add_embedding('prov', feature_conf.get('uprov_id').size, 16)
    embedding_layer.add_embedding('city', feature_conf.get('ucity_id').size, 16)
    embedding_layer.add_embedding('game_type', feature_conf.get('gt_id').size, 32)

    embedding_layer.add_embedding('uid_hs1', feature_conf.get('uid_hs1').size, 16)
    embedding_layer.add_embedding('uid_hs2', feature_conf.get('uid_hs2').size, 16)

    embedding_layer.add_embedding('aid_hs1', feature_conf.get('aid_hs1').size, 16)
    embedding_layer.add_embedding('aid_hs2', feature_conf.get('aid_hs2').size, 16)

    embedding_layer.add_embedding('a_info', feature_conf.get('a_info').size, 16)
    embedding_layer.add_embedding('a_active', feature_conf.get('a_active').size, 16)
    embedding_layer.add_embedding('a_gain', feature_conf.get('a_gain').size, 16)
    embedding_layer.add_embedding('a_hot', feature_conf.get('a_hot').size, 16)

    embedding_layer.add_embedding('u_info', feature_conf.get('u_info').size, 16)
    embedding_layer.add_embedding('u_active', feature_conf.get('u_active').size, 16)
    embedding_layer.add_embedding('u_pay', feature_conf.get('u_pay').size, 16)

    embedding_layer.add_embedding('crx_gend', feature_conf.get('x_gend').size, 8)
    embedding_layer.add_embedding('dwell_embed', feature_conf.get('uwtcaid7d_dr_bk').size, 8)
    embedding_layer.add_embedding('pay_embed', feature_conf.get('upayaid7d_cost_bk').size, 8)

    embedding_layer.add_embedding('uwtcaid7d_lts_bk', feature_conf.get('uwtcaid7d_lts_bk').size, 8)   #新增时效性特征对应的embedding
    # embedding_layer.add_embedding('weekday', feature_conf.get('weekday').size, 8)
    embedding_layer.add_embedding('hour', feature_conf.get('hour').size, 8)

    # 2.2 Concat Input Layer是deep model mlp网络的直接输入层
    concat_layer = ConcatLayer()
    # lookup->sum pooling的embedding特征
    concat_layer.add_dense(embedding_layer.lookup_pooling('prov', features.get('uprov_id'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('prov', features.get('aprov_id'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('city', features.get('ucity_id'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('city', features.get('acity_id'), 'sum'))  # 16

    concat_layer.add_dense(embedding_layer.lookup_pooling('crx_gend', features.get('x_gend'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('game_type', features.get('x_gt'), 'sum'))  # 32

    concat_layer.add_dense(embedding_layer.lookup_pooling('a_info', features.get('a_info'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('a_active', features.get('a_active'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('a_gain', features.get('a_gain'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('u_info', features.get('u_info'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('u_active', features.get('u_active'), 'sum'))  # 16
    concat_layer.add_dense(embedding_layer.lookup_pooling('u_pay', features.get('u_pay'), 'sum'))  # 16

    concat_layer.add_dense(embedding_layer.lookup_pooling('a_hot', features.get('a_hot'), 'sum'))  # 16

    # attention相关的embedding特征
    emb_gametype = embedding_layer.lookup('game_type', features.get('gt_id'))
    concat_layer.add_dense(embedding.pooling(emb_gametype, 'sum'))  # 32

    emb_uid_hs1 = embedding_layer.lookup('uid_hs1', features.get('uid_hs1'))
    concat_layer.add_dense(embedding.pooling(emb_uid_hs1, 'sum'))  # 16
    emb_uid_hs2 = embedding_layer.lookup('uid_hs2', features.get('uid_hs2'))
    concat_layer.add_dense(embedding.pooling(emb_uid_hs2, 'sum'))  # 16

    emb_aid_hs1 = embedding_layer.lookup('aid_hs1', features.get('aid_hs1'))
    concat_layer.add_dense(embedding.pooling(emb_aid_hs1, 'sum'))  # 16
    emb_aid_hs2 = embedding_layer.lookup('aid_hs2', features.get('aid_hs2'))
    concat_layer.add_dense(embedding.pooling(emb_aid_hs2, 'sum'))  # 16
    emb_combine_aid = tf.concat([emb_aid_hs1, emb_aid_hs2], -1)

    emb_ufavgt30d = embedding_layer.lookup('game_type', features.get('ufavgt30d_id'))  # 32
    emb_ufavgt30d = tf.multiply(emb_ufavgt30d, attention.weight(emb_ufavgt30d, emb_gametype, 32))
    concat_layer.add_dense(embedding.pooling(emb_ufavgt30d, 'sum'))  # 32

    emb_ufavaid30d_hs1 = embedding_layer.lookup('aid_hs1', features.get('ufavaid30d_hs1'))
    emb_ufavaid30d_hs2 = embedding_layer.lookup('aid_hs2', features.get('ufavaid30d_hs2'))
    emb_ufavaid30d = tf.concat([emb_ufavaid30d_hs1, emb_ufavaid30d_hs2], -1)
    emb_ufavaid30d = tf.multiply(emb_ufavaid30d, attention.weight(emb_ufavaid30d, emb_combine_aid, 32))
    concat_layer.add_dense(embedding.pooling(emb_ufavaid30d, 'sum'))  # 32

    emb_uwtcaid7d_hs1 = embedding_layer.lookup('aid_hs1', features.get('uwtcaid7d_hs1'))
    emb_uwtcaid7d_hs2 = embedding_layer.lookup('aid_hs2', features.get('uwtcaid7d_hs2'))
    emb_uwtcaid7d = tf.concat([emb_uwtcaid7d_hs1, emb_uwtcaid7d_hs2], -1)

    emb_uwtcaid7d_gt = embedding_layer.lookup('game_type', features.get('uwtcaid7d_gt_id'))   #用户播放序列对应game_type序列
    emb_aidgt = tf.concat([emb_combine_aid, emb_gametype],-1)       #合成aid特征：aid-gt, concat
    emb_aidgt_seq = tf.concat([emb_uwtcaid7d, emb_uwtcaid7d_gt],-1)

    dwell_embed_7d = embedding_layer.lookup('dwell_embed', features.get('uwtcaid7d_dr_bk'))
    lts_embed_7d = embedding_layer.lookup('uwtcaid7d_lts_bk', features.get('uwtcaid7d_lts_bk'))  # 得到时效性特征embedding向量
    wtc_attention = attention.weight(emb_aidgt_seq, emb_aidgt, 32, lts_embed_7d)
    uwtcaid7d_dur_w = tf.multiply(dwell_embed_7d, wtc_attention)    # 将时效性特征信息融入用户点击序列和目标交互的attention网络当中，将计算得到的注意力权重乘到时长的embedding上面
    emb_aidgt_seq = tf.multiply(emb_aidgt_seq, wtc_attention)       # 将时效性特征信息融入用户点击序列和目标交互的attention网络当中，计算得到的注意力权重乘到用户序列embedding上面
    concat_layer.add_dense(embedding.pooling(uwtcaid7d_dur_w, 'average', features.get('uwtcaid7d_msk')))  # 8
    concat_layer.add_dense(embedding.pooling(emb_aidgt_seq, 'average', features.get('uwtcaid7d_msk')))  # 32

    # 付费行为序列
    emb_upayaid7d_hs1 = embedding_layer.lookup('aid_hs1', features.get('upayaid7d_hs1'))
    emb_upayaid7d_hs2 = embedding_layer.lookup('aid_hs2', features.get('upayaid7d_hs2'))
    emb_upayaid7d = tf.concat([emb_upayaid7d_hs1, emb_upayaid7d_hs2], -1)
    pay_embed_7d = embedding_layer.lookup('pay_embed', features.get('upayaid7d_cost_bk'))
    pay_embed_vw = tf.multiply(pay_embed_7d, attention.weight(emb_upayaid7d, emb_combine_aid, 32))
    concat_layer.add_dense(embedding.pooling(pay_embed_vw, 'average', features.get('upayaid7d_msk')))  # 8
    concat_layer.add_dense(embedding.pooling(emb_upayaid7d, 'average', features.get('upayaid7d_msk')))  # 32

    #输入层concat星期几，第几小时特征
    # concat_layer.add_dense(embedding_layer.lookup_pooling('weekday', features.get('weekday'), 'sum'))  # 8
    concat_layer.add_dense(embedding_layer.lookup_pooling('hour', features.get('hour'), 'sum'))  # 8

    oh_follow15d = embedding.one_hot(features.get('x_isfollow'), feature_conf.get('x_isfollow').size)
    concat_layer.add_dense(embedding.pooling(oh_follow15d,'sum'))  #拼接关注标签特征

    hidden = concat_layer.forward()  # 496
    sharedExpertGroup = ExpertGroup(hidden, expert_units, task_num, mode)
    hidden_cross = build_cross_layers(hidden, 2, 0)

    # 计算各个任务的gate值
    clickGate = Gate(hidden, gate_units, task_num, mode)
    Gate1mins = Gate(hidden, gate_units, task_num, mode)
    Gate5mins = Gate(hidden, gate_units, task_num, mode)
    payGate = Gate(hidden, gate_units, task_num, mode)

    clickTowerInput = tf.reduce_sum(tf.multiply(tf.reshape(sharedExpertGroup, [-1, task_num, expert_units[-1]]), tf.expand_dims(clickGate, -1)),1)
    TowerInput1mins = tf.reduce_sum(tf.multiply(tf.reshape(sharedExpertGroup, [-1, task_num, expert_units[-1]]), tf.expand_dims(Gate1mins,-1)),1)
    TowerInput5mins = tf.reduce_sum(tf.multiply(tf.reshape(sharedExpertGroup, [-1, task_num, expert_units[-1]]), tf.expand_dims(Gate5mins,-1)),1)
    payTowerInput = tf.reduce_sum(tf.multiply(tf.reshape(sharedExpertGroup, [-1, task_num, expert_units[-1]]), tf.expand_dims(payGate,-1)),1)

    clickTowerInput = tf.concat([clickTowerInput, hidden_cross], -1)
    TowerInput1mins = tf.concat([TowerInput1mins, hidden_cross], -1)
    TowerInput5mins = tf.concat([TowerInput5mins, hidden_cross], -1)
    payTowerInput = tf.concat([payTowerInput, hidden_cross], -1)

    # 计算各个tower网络输出
    clickTowerOutput = Tower(clickTowerInput, tower_units, mode)
    click_out = tf.layers.dense(name='click_ctr', inputs=clickTowerOutput,
                                units=1, activation=None, use_bias=False,
                                kernel_initializer=tf.glorot_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001),
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)  # shape=[-1, 1]
    TowerOutput1mins = Tower(TowerInput1mins, tower_units, mode)
    out1mins = tf.layers.dense(name='1mins_ctr', inputs=TowerOutput1mins,
                                units=1, activation=None, use_bias=False,
                                kernel_initializer=tf.glorot_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001),
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)  # shape=[-1, 1]
    TowerOutput5mins = Tower(TowerInput5mins, tower_units, mode)
    out5mins = tf.layers.dense(name='5mins_ctr', inputs=TowerOutput5mins,
                                 units=1, activation=None, use_bias=False,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001),
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)  # shape=[-1, 1]
    payTowerOutput = Tower(payTowerInput, tower_units, mode)
    pay_out = tf.layers.dense(name='pay_ctr', inputs=payTowerOutput,
                              units=1, activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001),
                              trainable=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope("out"):
        bias_clk = tf.get_variable(name='bias_clk', shape=[1], initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32)
        ctr_out = lr_position_out + lr_out + click_out + bias_clk
        # ctr_out = lr_out + click_out + bias_clk
        ctr_prop = tf.sigmoid(ctr_out, name='ctr_prop')

        bias_1mins = tf.get_variable(name='bias_1mins', shape=[1], initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32)
        cvr_1out1mins = lr_out + out1mins + bias_1mins
        cvr_1mins_prop = tf.sigmoid(cvr_1out1mins, name='1mins_prop')

        bias_5mins = tf.get_variable(name='bias_5mins', shape=[1], initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
        cvr_out5mins = lr_out + out5mins + bias_5mins
        cvr_5mins_prop = tf.sigmoid(cvr_out5mins, name='5mins_prop')

        bias_pay = tf.get_variable(name='bias_pay', shape=[1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        cvr_pay_out = lr_out + pay_out + bias_pay
        cvr_pay_prop = tf.sigmoid(cvr_pay_out, name='pvr_prop')

        ctcvr_1mins_prop = ctr_prop * cvr_1mins_prop
        ctcvr_5mins_prop = ctr_prop * cvr_5mins_prop
        ctcvr_pay_prop = ctr_prop * cvr_pay_prop


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"ctr_prop": ctr_prop, "cvr_1mins_prop": cvr_1mins_prop, "cvr_5mins_prop": cvr_5mins_prop, "cvr_pay_prop": cvr_pay_prop}
        export_outputs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    ctr_label = labels.get("clk")
    mins_label = labels.get("is_1min")
    clk_w = labels.get("clk_w")
    ctcvr_5mins_label = labels.get("is_5min")
    ctcvr_pay_label = labels.get("pay")
    ctcvr_pay_weight = labels.get("pay_w")

    ctr_sample_loss = clk_w * tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_out)
    #ctr_sample_loss = focal_loss(labels=ctr_label, predictions=ctr_prop, weight=clk_w, alpha=0.25, gamma=2)
    ctcvr_1mins_sample_loss = logloss(weight=1, labels=mins_label, predictions=ctcvr_1mins_prop)
    #ctcvr_5minsample_loss = focal_loss(labels=mins_label, predictions=ctcvr_5mins_prop, weight=1.0, alpha=0.25, gamma=2)
    ctcvr_5mins_sample_loss = logloss(weight=1, labels=ctcvr_5mins_label, predictions=ctcvr_5mins_prop)
    #ctcvr_follow_sample_loss = focal_loss(labels=ctcvr_follow_label, predictions=ctcvr_follow_prop, weight=1.0, alpha=0.25, gamma=2)
    ctcvr_pay_sample_loss = logloss(weight=ctcvr_pay_weight, labels=ctcvr_pay_label, predictions=ctcvr_pay_prop)
    #ctcvr_pay_sample_loss = focal_loss(labels=ctcvr_pay_label, predictions=ctcvr_pay_prop, weight=ctcvr_pay_weight, alpha=0.25, gamma=2)

    auc = {}
    auc["auc_ctr"] = tf.metrics.auc(ctr_label, ctr_prop)
    auc["auc_cvr_1mins"] = tf.metrics.auc(mins_label, cvr_1mins_prop)
    # auc["auc_ctcvr_5mins"] = tf.metrics.auc(mins_label, ctcvr_1mins_prop)
    auc["auc_cvr_5mins"] = tf.metrics.auc(ctcvr_5mins_label, cvr_5mins_prop)
    # auc["auc_ctcvr_follow"] = tf.metrics.auc(ctcvr_follow_label, ctcvr_follow_prop)
    auc["auc_cvr_pay"] = tf.metrics.auc(ctcvr_pay_label, cvr_pay_prop)
    # auc["auc_ctcvr_pay"] = tf.metrics.auc(ctcvr_pay_label, ctcvr_pay_prop)

    loss_sample = ctr_sample_loss + ctcvr_1mins_sample_loss + ctcvr_5mins_sample_loss + ctcvr_pay_sample_loss
    loss = tf.reduce_mean(loss_sample) + tf.losses.get_regularization_loss()


    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric = auc
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([optimizer.minimize(loss, global_step=tf.train.get_global_step()), update_ops])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)


def input_fn(file_names, column_dic, label_meta, batch_size=512, num_epochs=1, shuffle_size=1024, num_paral=4,
             prefetch_size=4096):
    parser = TFRecordParser(column_dic, label_meta)
    features, labels = parser.iterator(file_names,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs,
                                       shuffle_size=shuffle_size,
                                       num_paral=num_paral,
                                       prefetch_size=prefetch_size)
    return features, labels


def main(_):
    config = tf.estimator.RunConfig()
    config = config.replace(log_step_count_steps=1000,
                            save_summary_steps=100,
                            save_checkpoints_steps=3000)
    if FLAGS.rng:
        config = config.replace(tf_random_seed=FLAGS.rng)  # 设置随机数：实验复现

    feature_conf, label_conf = load_sample_spec(FLAGS.feature_spec)  # 特征配置 & 目标配置
    expert_units = list(map(int, FLAGS.expert_units.split(',')))
    gate_units = list(map(int, FLAGS.gate_units.split(',')))
    gate_num = FLAGS.gate_num
    tower_units = list(map(int, FLAGS.tower_units.split(',')))
    task_num = FLAGS.task_num
    dcn_num = FLAGS.dcn_num

    model_paras = {
        'feature_conf': feature_conf,
        'expert_units': expert_units,
        'label_conf': label_conf,
        'dcn_num': dcn_num,
        'gate_units': gate_units,
        'gate_num': gate_num,
        'tower_units': tower_units,
        'task_num': task_num
    }
    checkpoint_dir = FLAGS.checkpoint  # checkpoints存储路径
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=checkpoint_dir, params=model_paras, config=config)

    # 训练模型
    if FLAGS.task == 'train':
        # 获取train文件列表
        tr_files = glob.glob("%s/train*.tfrecord" % FLAGS.data)
        if not FLAGS.rng:
            random.shuffle(tr_files)
        print("train_files:", tr_files)

        # 获取validate文件列表
        va_files = glob.glob("%s/validate*.tfrecord" % FLAGS.data)
        if not va_files:
            va_files = glob.glob("%s/test*.tfrecord" % FLAGS.data)
        print("validate_files:", va_files)

        # try:
        #     shutil.rmtree(checkpoint_dir)
        # except Exception as e:
        #     print(e, "at clear_model")
        # else:
        #     print("start new learning task -> existing neuralnet cleaned at %s" % checkpoint_dir)

        # train
        print('epoch={},batch size={},learning rate={}'.format(FLAGS.epoch, FLAGS.batch, FLAGS.learning_rate))
        tf.estimator.train_and_evaluate(model,
                                        tf.estimator.TrainSpec(
                                            input_fn=lambda: input_fn(tr_files, feature_conf, label_conf,
                                                                      num_epochs=FLAGS.epoch,
                                                                      batch_size=FLAGS.batch)),
                                        tf.estimator.EvalSpec(
                                            input_fn=lambda: input_fn(va_files, feature_conf, label_conf,
                                                                      num_epochs=1,
                                                                      batch_size=FLAGS.batch,
                                                                      shuffle_size=0),
                                            steps=4000, start_delay_secs=30, throttle_secs=60))

    # 导出模型
    if FLAGS.task == 'export':
        # checkpoint导出pb模型
        export_dir = FLAGS.saved_model
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir)
        os.makedirs(export_dir)

        feature_spec = build_feature_spec(feature_conf)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_saved_model(export_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
