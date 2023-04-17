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

def gene_theo_ions(prot_seq, acetylation):
  left_ions = []
  right_ions = []

  acetylation_weight = 42.0106
  weights = {'A': 71.03711, 'C': 103.00919, 'D': 115.02694, 'E': 129.04259, 'F': 147.06841, 'G': 57.02146, 
    'H': 137.05891, 'I': 113.08406, 'K': 128.09496, 'L': 113.08406, 'M': 131.04049, 'N': 114.04293,
    'P': 97.05276, 'Q': 128.05858, 'R': 156.10111, 'S': 87.03203, 'T': 101.04768, 'V': 99.06841,
    'W': 186.07931, 'Y': 163.06333}

  if acetylation: 
    left_ions.append(weights[prot_seq[0]] + acetylation_weight) 
  else:
    left_ions.append(weights[prot_seq[0]])
  right_ions.append(weights[prot_seq[len(prot_seq)-1]])

  for idx in range(1, len(prot_seq)):
    left_ions.append(left_ions[idx -1] + weights[prot_seq[idx]])
    right_ions.append(right_ions[idx -1] + weights[prot_seq[len(prot_seq)-1-idx]])

  return left_ions, right_ions

def get_modified_fragments(mass_list, shift):
  mod_mass_list = [x + shift for x in mass_list]
  return mod_mass_list
