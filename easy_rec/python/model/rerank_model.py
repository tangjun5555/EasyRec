# -*- encoding:utf-8 -*-

import logging
import tensorflow as tf

from easy_rec.python.model.easy_rec_model import EasyRecModel


if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class ReRankModel(EasyRecModel):
    def __init__(self,
                 model_config,
                 feature_configs,
                 features,
                 labels=None,
                 is_training=False
                 ):
        super(ReRankModel, self).__init__(model_config, feature_configs, features, labels, is_training)
        self._loss_type = self._model_config.loss_type
