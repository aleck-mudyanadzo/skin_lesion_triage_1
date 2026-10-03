@echo off
REM ============================================================
REM setup.bat - Windows setup script for the Skin Lesion Triage System
REM Run this from the project root: double-click, or `setup.bat` in cmd.
REM ============================================================

echo [*] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.10-3.12 from python.org
    echo and make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ('python -c "import sys; print(sys.version_info.major); print(sys.version_info.minor)" 2^>nul') do (
    if not "%%a"=="3" goto :skip_version_check
    if "%%b" GEQ "10" if "%%b" LEQ "12" goto :version_ok
)
:skip_version_check
    echo ERROR: Python 3.10-3.12 is required for this project.
    echo Please install Python 3.12 (recommended) or 3.10-3.12 from python.org.
    echo Then rerun setup.bat.
    pause
    exit /b 1
:version_ok

echo [*] Creating virtual environment (venv)...
python -m venv venv

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo [*] Creating required folders...
if not exist "instance" mkdir instance
if not exist "models" mkdir models
if not exist "app\static\uploads" mkdir app\static\uploads
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed

echo.
echo ============================================================
echo Setup complete!
echo.
echo Next steps:
echo   1. Get HAM10000: python scripts\prepare_data.py --kaggle_download
echo   2. Train a model (ideally on Google Colab for GPU speed):
echo      python scripts\train_model.py --arch mobilenetv2
echo   3. Copy the trained .keras file into the models\ folder
echo   4. Run the app: python run.py
echo   5. Open http://127.0.0.1:5000 in your browser
echo ============================================================
pause
