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

import os
import sys
from EnvCNN.Data.feature_reader import read_feature_file
from EnvCNN.Data.matrix_writer import write as matrix_writer

anno_file = sys.argv[1]
file_prefix = os.path.basename(os.getcwd())
env_list_with_features = read_feature_file(anno_file)
for env in env_list_with_features:
  mat = env.getMatrix()
  matrix_writer(env, mat, file_prefix)
