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

import math

def shortlist_envelopes_by_prsm_id(prsm, envelope_list):
  shortlisted_envelopes = []
  for env in envelope_list:
    if env.header.spec_id == prsm.spec_id:
      shortlisted_envelopes.append(env)
  return shortlisted_envelopes

def shortlist_envelopes_by_charge(envelope_list, charge):
  shortlisted_envelopes = []
  for env in envelope_list:
    if env.header.charge == charge:
      shortlisted_envelopes.append(env)
  return shortlisted_envelopes

def assign_labels_by_peak_id(prsm, envelope_list):
  for env in envelope_list:
    if env.header.peak_id in prsm.matched_peaks_id:
      env.header.label = 1
      env.header.ion_type = 'File'

def assign_labels(env):
  if env.header.ion_type == 'B' or env.header.ion_type == 'Y':
    env.header.label = 1

def get_rounded_value(value):
  round_param = 10**2
  return math.floor(value * round_param)/round_param

def get_rounded_mass(peak_list):
  """ Round off to 2 decimal places"""
  rounded_masses = []
  for peak in peak_list:
    rounded_masses.append(get_rounded_value(peak.mass))
  return rounded_masses

def get_normalized_intensity(peak_intensity, normalization_factor):
  """ Normalize using highest theoretical envelope intensity"""
  normalized_intens = peak_intensity/normalization_factor
  return normalized_intens

def get_max_intensity(peak_list):
  max_intensity = max(peak_list, key=lambda x: x.intensity).intensity
  return max_intensity

def get_bin_indx(mass, minimum_mz):
  bin_index = int((mass - minimum_mz)* 100)
  return bin_index

def find_exp_peak_idx(peak, exp_peak_list, tolerance):
  exp_peak_idx = -1
  thr_mass = peak.mass
  old_mass_diff = math.inf
  for exp_idx in range(0, len(exp_peak_list)):
    exp_mass = exp_peak_list[exp_idx].mass
    mass_diff = thr_mass - exp_mass
    if mass_diff <= tolerance and mass_diff >= -tolerance:
      if (abs(mass_diff) < abs(old_mass_diff)):
        exp_peak_idx = exp_idx
        old_mass_diff = mass_diff
  return exp_peak_idx

def populate_matrix(feature, matrix, mono_mass):
  pos = feature.bin_index
  if pos > -1 and pos < 300:
    matrix[pos][0] = feature.norm_theo_peak_inte
    matrix[pos][1] = feature.norm_exp_peak_inte
    mass_diff = abs(feature.mass_diff)
    if mass_diff == 0:
      matrix[pos][2] = 1
    elif mass_diff > 0.02:
      matrix[pos][2] = 0
    else:
      matrix[pos][2] = (0.02 - mass_diff)/0.02
    matrix[pos][3] = feature.inte_diff
    matrix[pos][4] = math.log10((10**feature.log_theo_intensity)/(10**feature.log_base_intenisty))
    matrix[pos][5] = math.log10(mono_mass)
