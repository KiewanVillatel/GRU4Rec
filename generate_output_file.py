# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys

sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation

PATH_TO_TRAIN = '../data/rsc15_train_full.txt'
PATH_TO_TEST = '../data/rsc15_test.txt'

if __name__ == '__main__':
  data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
  valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})

  # State-of-the-art results on RSC15 from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
  # BPR-max, no embedding (R@20 = 0.7197, M@20 = 0.3157)
  gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                        n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                        momentum=0.1, n_sample=2048, sample_alpha=0, bpreg=0.5, constrained_embedding=True)
  gru.fit(data)
  res = evaluation.evaluate_gpu(gru, valid, output_path='predictions_GRU4REC.csv', mode='standard')
  print('Recall@20: {}'.format(res[0]))
  print('MRR@20: {}'.format(res[1]))