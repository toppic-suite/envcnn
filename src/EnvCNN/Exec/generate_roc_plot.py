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
import EnvCNN.Data.test_model_util as test_model_util 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

anno_dir = sys.argv[1]
files = os.listdir(anno_dir)

labels = []
pred_score = []
topfd_score = []
for anno_file in files:
  env_list = read_anno_file(os.path.join(anno_dir, anno_file))
  env_label = []
  env_pred_score = []
  env_topfd_score = []
  for env in env_list:
    env_label.append(env.header.label)
    env_pred_score.append(env.header.pred_score)
    env_topfd_score.append(env.header.topfd_score)
  
  max_topfd_score = max(env_topfd_score)
  if max_topfd_score > 0:
    normalized_env_topfd_score = [x / max_topfd_score for x in env_topfd_score]
  else:
    normalized_env_topfd_score =  env_topfd_score
  
  labels.extend(env_label)
  pred_score.extend(env_pred_score)
  topfd_score.extend(normalized_env_topfd_score)

test_model_util.generate_roc_curve(os.getcwd(), pred_score, topfd_score, labels)
