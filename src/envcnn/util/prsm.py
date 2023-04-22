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

import EnvCNN.Data.prot_seq as ProtSeq

class Prsm:
  """This class contains protein spectral match data.
  Moreover, this class is utilized from PrsmList class
  """
  def __init__(self, spec_id, prot_sequence, peak_list, acetylation):
    self.spec_id = spec_id
    self.prot_sequence = prot_sequence
    self.peak_list = peak_list
    self.acetylation = acetylation

  def add_annotation(self, env_list, mass_list, shift, anno):
    frags = ProtSeq.get_modified_fragments(mass_list, shift)

    for env in env_list:
      peak_mass = env.header.mono_mass
      for j in range(len(frags)):
        frag_mass = frags[j]
        tol = (15 * frag_mass) / 1e6
        if (tol < 0.01):
          tol = 0.01
        if (abs(peak_mass - frag_mass) <= tol):
          if env.header.ion_type == "":
            env.header.ion_type = anno
          break

  def annotate(self, env_list):
    nterm_masses, cterm_masses = ProtSeq.gene_theo_ions(self.prot_sequence, self.acetylation)
    Proton = 1.00727647
    H = 1.007825035
    O = 15.99491463
    CO = 12.0000 + O
    NH3 = 14.003074 + H + H + H
    H2O = H + H + O

    # b -ion
    self.add_annotation(env_list, nterm_masses, 0.0, "B")
    # y -ion
    self.add_annotation(env_list, cterm_masses, 19.0184-Proton, "Y")
    # c - ion
    self.add_annotation(env_list, nterm_masses, 18.0344-Proton, "C")
    # z' - ion
    self.add_annotation(env_list, cterm_masses, 1.9918-Proton+Proton, "Z+1")
    # a -ion
    self.add_annotation(env_list, nterm_masses, -CO, "A")
    # x -ion
    self.add_annotation(env_list, cterm_masses, CO, "X")
    # b - water
    self.add_annotation(env_list, nterm_masses, -H2O, "B-Water")
    # y - water
    self.add_annotation(env_list, cterm_masses, 19.0184-Proton-H2O, "Y-Water")
    # b - ammonia
    self.add_annotation(env_list, nterm_masses, -NH3, "B-Ammonia")
    # y - ammonia
    self.add_annotation(env_list, cterm_masses, 19.0184-Proton-NH3, "Y-Ammonia")
    # b - 1
    self.add_annotation(env_list, nterm_masses, -Proton, "B-1")
    # y - 1
    self.add_annotation(env_list, cterm_masses, 19.0184-Proton-Proton, "Y-1")
    # b + 1
    self.add_annotation(env_list, nterm_masses, Proton, "B+1")
    # y + 1
    self.add_annotation(env_list, cterm_masses, 19.0184, "Y+1")

