@echo off
echo ============================================================
echo      Image Caption Generator - Windows Environment Setup
echo ============================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.11+ and try again.
    goto :end
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :end
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Skipping creation.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing PyTorch with RTX GPU CUDA 12.1 support...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    goto :end
)

echo.
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    goto :end
)

echo.
echo Running verification script...
python src/utils/check_gpu.py

:end
echo.
echo Press any key to exit...
pause >nul
