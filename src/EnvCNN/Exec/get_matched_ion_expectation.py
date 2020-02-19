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

import sys
import os
from EnvCNN.Data.feature_reader import read_feature_file
from EnvCNN.Data.prsm_reader import read_prsm_file 
import numpy as np
import matplotlib.pyplot as plt

feature_dir = sys.argv[1] 
prsm_dir = sys.argv[1]
file_list = os.listdir(feature_dir)

anno = ["B", "Y", "C", "Z+1", "A", "X", "B-Water", "Y-Water", "B-Ammonia", "Y-Ammonia", "B-1", "Y-1", "B+1", "Y+1", ""]
d = dict.fromkeys(anno, 0)
TotalExpectation = 0
TotalTheo = 0anno = ["B", "Y", "C", "Z+1", "A", "X", "B-Water", "Y-Water", "B-Ammonia", "Y-Ammonia", "B-1", "Y-1", "B+1", "Y+1", ""]
d = dict.fromkeys(anno, 0)
TotalExpectation = 0
TotalTheo = 0
for data_file in file_list:
  feature_file = os.path.join(feature_dir, data_file)
  sp_file = data_file.replace("feature", "sp")
  index_dot = sp_file.rindex(".")
  sp_file = sp_file[0:index_dot]
  prsm_file = os.path.join(prsm_dir, sp_file + ".xml")
  env_list_with_features = read_feature_file(feature_file)
  prsm_data = read_prsm_file(prsm_file)
  squence_length = len(prsm_data.prot_sequence)
  M = prsm_data.mono_mass
  X = 0
  for peak in prsm_data.peak_list:
    tol = peak.mass * (15/1E6)
    X = X + tol
  Expectation = (X/M)*(squence_length-1)
  TotalTheo = TotalTheo  + (squence_length-1)
  TotalExpectation = TotalExpectation + (2*Expectation)
  for env in env_list_with_features:
    d[env.header.ion_type] = d[env.header.ion_type] + 1

print("Expectation: ", TotalExpectation)

vals = list(d.values())
keys = list(d.keys())
percent = [100*(1- (TotalExpectation/v)) for v in vals]
FDR = [100-p for p in percent]
## plot FDR and labels frequency
labels = ['b-ion', 'y-ion', 'c-ion', 'z-ion + H', 'a-ion', 'x-ion', 'b-ion - H$_{2}$O', 'y-ion - H$_{2}$O', 'b-ion - NH$_{3}$', 'y-ion - NH$_{3}$', 'b-ion - H', 'y-ion - H', 'b-ion + H', 'y-ion + H', '']
y_pos = np.arange(len(keys) - 1)

plt.figure()
plt.bar(y_pos, vals[0:len(vals)-1], align='center', alpha=0.5)
plt.xticks(y_pos, labels, rotation=35, fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel('Occourance')
plt.savefig("matched_ions_occurance.png", dpi=1500)
plt.close()

plt.figure()
plt.bar(y_pos, FDR[0:len(vals)-1], align='center', alpha=0.5)
plt.xticks(y_pos, labels, rotation=35, fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel('FDR (%)')
plt.savefig("matched_ions_FDR.png", dpi=1500)
plt.close()