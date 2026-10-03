@echo off
setlocal

REM Run the CADOPT Polyscope UI app inside the cad-env conda environment.
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "CONDA_EXE=C:\Users\ribin\miniconda3\Scripts\conda.exe"

if not exist "%CONDA_EXE%" (
  echo [ERROR] conda.exe not found at:
  echo         %CONDA_EXE%
  echo Update CONDA_EXE in run_ui.bat to your Miniconda/Anaconda location.
  pause
  exit /b 1
)

if not exist "Demo\mechanical_assembly.stp" (
  echo [WARN] Demo\mechanical_assembly.stp not found.
  echo The app loads the first .stp/.step file from .\Demo\ when you click "1. Load ^& Tessellate".
)

echo Starting CAD-Anchored Field Remesher UI...
"%CONDA_EXE%" run -n cad-env python main.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] App exited with code %EXIT_CODE%.
)

pause
exit /b %EXIT_CODE%
