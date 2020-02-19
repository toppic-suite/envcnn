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

#!/usr/bin/env python3

import sys
import os
import re
import numpy
from EnvCNN.Data.feature_reader import read_feature_file
from EnvCNN.Data.env_writer import write
from keras.models import load_model

anno_dir = sys.argv[1]
model_dir = sys.argv[2]
model = load_model(os.path.join(model_dir, "model.h5"))

#files = [f for f in os.listdir(anno_dir) if re.match(r'feature*', f)]
files = [f for f in os.listdir(anno_dir) if re.match(r'*.env', f)]

dir_name = "output_envs"
ouput_dir = os.path.join(model_dir, dir_name)
if os.path.isdir(ouput_dir) == False:
  os.mkdir(ouput_dir)
	
for anno_file in files:
  env_list_with_features = read_feature_file(os.path.join(anno_dir, anno_file))
  for env in env_list_with_features:
    mat = env.getMatrix()
    b = mat[numpy.newaxis,:, :]
    pred_score = model.predict(b)
    env.header.pred_score = pred_score[0][0]
  
  env_list_with_features.sort(key=lambda x: x.header.pred_score, reverse=True)
  output_file_name = os.path.join(ouput_dir, "Envelope_" + str(env.header.spec_id) + ".env")
  write(env_list_with_features, output_file_name)
