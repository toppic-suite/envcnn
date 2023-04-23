#!/usr/bin/env python3

import os
import sys
from util.env_header import EnvHeader
from util.feature_reader import read_feature_file

env_file_name = sys.argv[1]
#output_file_name = sys.argv[2]
env_list_with_features = read_feature_file(env_file_name)
for env in env_list_with_features:
  env_info = env.header.info()
  print(env_info)
