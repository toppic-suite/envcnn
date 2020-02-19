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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

topfd_prsm_file = sys.argv[1]
envcnn_prsm_file = sys.argv[2]

topfd = pd.read_csv(topfd_prsm_file, skiprows=23)
model = pd.read_csv(envcnn_prsm_file, skiprows=23)

topfd_spec_id = list(topfd.iloc[:,2])
model_spec_id = list(model.iloc[:,2])
common_ids = list(set(model_spec_id).intersection(set(topfd_spec_id)))

## Model
frag_diff = []
for spec in common_ids:
  td = topfd.loc[topfd['Spectrum ID'] == spec]
  md = model.loc[model['Spectrum ID'] == spec]
  td_matched_peaks = int(td['#matched peaks'])
  md_matched_peaks = int(md['#matched peaks'])
  frag_diff.append(md_matched_peaks - td_matched_peaks)

d = {x:frag_diff.count(x) for x in frag_diff}
d = dict(sorted(d.items()))
a, b = d.keys(), d.values()

graph_file_name = os.path.join(os.getcwd(), "Matched_masses_difference_frequency.png")
plt.figure()
plt.bar(a, b)
plt.xlabel('Difference between the matched EnvCNN and MS-Deconv masses')
plt.ylabel('Frequency of PrSMs')
plt.savefig(graph_file_name, dpi=1500)
plt.close()
