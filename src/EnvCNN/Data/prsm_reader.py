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

import xml.etree.ElementTree as ET
from EnvCNN.Data.anno_peak import AnnoPeak
from EnvCNN.Data.prsm import Prsm

def read_prsm_file(filename):
  """Reads data from the file provided in filename and calls __init__ function"""
  tree = ET.parse(filename)
  number_of_modifications = int(_get_value(tree,'./annotated_protein/unexpected_change_number'))
  if number_of_modifications > 0:
    print("prsm_reader.py: ERROR! The Prsm has unknown modifications!")
    return
  # Extract data from xml file and assign to the class object
  spec_id = int(_get_value(tree, './ms/ms_header/ids'))
  peak_list = _get_match_peak_list(tree)
  start_position = int(_get_value(tree, './annotated_protein/annotation/first_residue_position'))
  end_position = int(_get_value(tree, './annotated_protein/annotation/last_residue_position'))
  acetylation = int(_get_value(tree,'./annotated_protein/n_acetylation'))
  prot_whole_seq = _get_prot_seq(tree)
  prot_seq = prot_whole_seq[start_position: end_position + 1]
  return Prsm(spec_id, prot_seq, peak_list, acetylation)

def _get_value(tree, address):
  sc = tree.find(address)
  if sc != None:
    return sc.text
  else:
    raise Exception("Error Loading Prsm Data")

def _get_match_peak_list(tree):
  peak_list =[]
  for matched_ion in tree.iter(tag = 'peak'):
    peak_id = int(_get_value(matched_ion, 'peak_id'))
    peak_mass = float(_get_value(matched_ion, 'monoisotopic_mass'))
    peak_inte = 0
    peak_anno = ""
    peak = AnnoPeak(peak_id, peak_mass, peak_inte, peak_anno)
    peak_list.append(peak)
  return peak_list

def _get_prot_seq(tree):
  seq = ""
  for matched_aa in tree.findall('./annotated_protein/annotation/residue'):
    amino_acid = _get_value(matched_aa, 'acid')
    seq = seq + amino_acid
  return seq
