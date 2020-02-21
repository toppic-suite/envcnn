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
from EnvCNN.Data.env_reader import read_env_file
from EnvCNN.Data.prsm_reader import read_prsm_file
from EnvCNN.Data.env_writer import write as env_writer
import EnvCNN.Data.env_util as env_util

env_file = sys.argv[1]
prsm_file  = sys.argv[2]

start = prsm_file.index('_') + 1
end = prsm_file.index( '.', start )
index = prsm_file[start:end]
output_file  = "annotated_" + prsm_file[start:end] + ".env"

env_list = read_env_file(env_file)
prsm = read_prsm_file(prsm_file)
if prsm is None:
  exit()

prsm.annotate(env_list)
for env in env_list:
  env_util.assign_labels(env)
  env.get_intv_peak_list()

env_writer(env_list, output_file)
