@echo off
set PY=.venv\Scripts\python.exe
%PY% -m pytest -q
if errorlevel 1 exit /b 1
%PY% run.py --mode demo
%PY% -m poisoning.build_training_data
%PY% run.py --mode experiment
