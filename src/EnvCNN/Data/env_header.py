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

class EnvHeader:
  def __init__(self, file_name, spec_id, 
      base_inte, peak_id, charge, mono_mz, mono_mass, inte_sum, 
      topfd_score, pred_score, ion_type, label):
    self.file_name = file_name
    self.spec_id = spec_id
    # noise signal intensity of the spectrum
    self.base_inte = base_inte
    self.peak_id = peak_id
    self.charge = charge
    self.mono_mz = mono_mz
    self.mono_mass = mono_mass
    self.inte_sum = inte_sum
    self.topfd_score = topfd_score
    self.pred_score = pred_score
    self.ion_type = ion_type
    self.label = label
  
  @classmethod
  def get_header(cls, spec_id, base_inte, peak_id,
      charge, mono_mz, mono_mass, inte_sum, topfd_score):
    file_name = ""
    pred_score = 0
    ion_type = ""
    label = 0
    return cls(file_name, spec_id, base_inte, peak_id,
    charge, mono_mz, mono_mass, inte_sum, topfd_score, pred_score,
    ion_type, label)

  @classmethod
  def get_header_with_anno(cls, spec_id, base_inte, peak_id,
      charge, mono_mz, mono_mass, inte_sum, topfd_score, ion_type, label, pred_score):
    file_name = ""
    return cls(file_name, spec_id, base_inte, peak_id,
    charge, mono_mz, mono_mass, inte_sum, topfd_score, pred_score,
    ion_type, label)