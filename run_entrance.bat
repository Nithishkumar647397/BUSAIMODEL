@echo off
REM ============================================================================
REM SmartBus TN - Entrance Counter Launcher
REM Edit the camera index below if your entrance camera uses a different port:
REM ============================================================================
set CAM=0

echo ============================================================================
echo  SmartBus TN - Starting ENTRANCE Counter (Camera %CAM%)
echo ============================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment 'venv' not found. Please create it first.
    pause
    exit /b 1
)

python counter.py --door entrance --camera %CAM%

echo.
echo ============================================================================
echo  Entrance counter session closed.
echo ============================================================================
pause
