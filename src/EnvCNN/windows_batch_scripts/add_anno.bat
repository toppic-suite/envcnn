@echo off
setlocal EnableDelayedExpansion

for %%S in (sp*.xml); do (
  SET VAR=%%S
  for /f "tokens=2 delims=_" %%A in ("!VAR!") do ( 
    for /f "tokens=1 delims=." %%B in ("%%A") do ( 
      SET NUM=%%B
      call :innerloop
    ) 
  ) 
)

:innerloop
for %%E in (*.env); do (
  echo "%%E" | findstr /i /c:"!NUM!" >nul
  if not errorlevel 1 (
    echo SHORT_VAL: %%E "!VAR!" "!NUM!"
    python ./../../src/EnvCNN/Exec/add_anno.py %%E !VAR! annotated_!NUM!.env
    goto :eof
  ) 
)

endlocal
