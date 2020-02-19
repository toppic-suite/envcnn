@echo off
setlocal EnableDelayedExpansion

@echo off

Set "original_extension=.xml"
Set "required_extension=.env"

for %%S in (sp_*.xml); do (
  Set "prsm_file=%%~S"
  SET env_file=!prsm_file:%original_extension%=%required_extension%!
  python ./../../src/EnvCNN/Exec/add_anno.py !env_file! !prsm_file!
)

endlocal
