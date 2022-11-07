##Copyright (c) 2014 - 2020, The Trustees of Indiana University.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.

#!/usr/bin/python3

import os
import sys
from EnvCNN.Data.anno_reader import read_anno_file
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

anno_dir = sys.argv[1]
files = os.listdir(anno_dir)

env_list_2d_topfd = []
for anno_file in files:
  env_list = read_anno_file(os.path.join(anno_dir, anno_file))
  ## Rank by TopFD score
  env_list.sort(key=lambda x: x.header.topfd_score, reverse=True)
  env_list_2d_topfd.append(env_list)
  
env_list_2d_pred_score = []
for anno_file in files:
  env_list = read_anno_file(os.path.join(anno_dir, anno_file))
  ## Rank by EnvCNN prediction score  
  env_list.sort(key=lambda x: x.header.pred_score, reverse=True)
  env_list_2d_pred_score.append(env_list)

## Computing Ranks TopFD
rank_len = 500
b_list_topfd = [0] * rank_len
y_list_topfd = [0] * rank_len
for i in range(len(env_list_2d_topfd)):
  for j in range(len(env_list_2d_topfd[i])):
    if j >= rank_len:
      break
    if env_list_2d_topfd[i][j].header.ion_type == "B":
      b_list_topfd[j] = b_list_topfd[j] + 1
    if env_list_2d_topfd[i][j].header.ion_type == "Y":
      y_list_topfd[j] = y_list_topfd[j] + 1

## Computing Ranks Pred_Score
rank_len = 500
b_list_pred_score = [0] * rank_len
y_list_pred_score = [0] * rank_len
for i in range(len(env_list_2d_pred_score)):
  #print("env_list", i)
  for j in range(len(env_list_2d_pred_score[i])):
    if j >= rank_len:
      break
    if env_list_2d_pred_score[i][j].header.ion_type == "B":
      b_list_pred_score[j] = b_list_pred_score[j] + 1
    if env_list_2d_pred_score[i][j].header.ion_type == "Y":
      y_list_pred_score[j] = y_list_pred_score[j] + 1

## Computing percentage
length = 100
pred_score_percentage = [0] * length
topFD_percentage = [0] * length
for i in range(length):
  pred_score_percentage[i] = (b_list_pred_score[i]+y_list_pred_score[i])
  topFD_percentage[i] = (b_list_topfd[i]+y_list_topfd[i])

## Plotting Graph  
graph_file_name = os.path.join(os.getcwd(), "Rank.png")
plt.figure()
plt.plot(list(range(0, length)), pred_score_percentage)
plt.plot(list(range(0, length)), topFD_percentage)
plt.title('Rank Plot')
plt.ylabel('Number of PrSMs with label 1')
plt.xlabel('Rank Number')
plt.legend(['Model', 'TopFD'], loc='upper right')
plt.savefig(graph_file_name, dpi=500)
plt.close()
