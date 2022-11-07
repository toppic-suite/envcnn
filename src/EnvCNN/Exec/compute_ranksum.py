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

anno_dir = sys.argv[1]
files = os.listdir(anno_dir)
file_num = len(files)

env_list_2d_topfd_score=[]
for anno_file in files:
  env_list = read_anno_file(os.path.join(anno_dir , anno_file))
  ## Sort by TopFD score
  env_list.sort(key=lambda x: x.header.topfd_score, reverse=True)
  env_list_2d_topfd_score.append(env_list)
  
env_list_2d_pred_score=[]
for anno_file in files:
  env_list = read_anno_file(os.path.join(anno_dir , anno_file))
  ## Sort by EnvCNN score
  env_list.sort(key=lambda x: x.header.pred_score, reverse=True)
  env_list_2d_pred_score.append(env_list)

ranksum = []
for i in range(len(env_list_2d_pred_score)):
  env_ranksum = 0
  for j in range(len(env_list_2d_pred_score[i])):
    if env_list_2d_pred_score[i][j].header.label == 1:
      env_ranksum = env_ranksum + (j + 1)
  ranksum.append(env_ranksum)
print("RankSUM value:", sum(ranksum))

topfd_ranksum = []
for i in range(len(env_list_2d_topfd_score)):
  env_ranksum = 0
  for j in range(len(env_list_2d_topfd_score[i])):
    if env_list_2d_topfd_score[i][j].header.label == 1:
      env_ranksum = env_ranksum + (j + 1)
  topfd_ranksum.append(env_ranksum)
print("TopFD RankSUM value:", sum(topfd_ranksum)) 
