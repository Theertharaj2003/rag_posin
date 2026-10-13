$ErrorActionPreference = "Stop"

py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Environment installed. Run:"
Write-Host "  .\.venv\Scripts\python.exe run.py --mode demo"
Write-Host "  .\.venv\Scripts\python.exe -m pytest -q"
