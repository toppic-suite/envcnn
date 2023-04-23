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

  def info(self):
    txt = "" 
    txt = txt + str(self.spec_id) + "_" + str(self.peak_id) + "\t"
    txt = txt + str(self.charge) + "\t"
    txt = txt + str(self.mono_mz) + "\t"
    txt = txt + str(self.mono_mass) + "\t"
    txt = txt + str(self.inte_sum) + "\t"
    txt = txt + self.ion_type + "\t"
    txt = txt + str(self.label)
    return txt
  
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
