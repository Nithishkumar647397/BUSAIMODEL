@echo off
REM ============================================================================
REM SmartBus TN - Exit Counter Launcher
REM Edit the camera index below if your exit camera uses a different port:
REM ============================================================================
set CAM=1

echo ============================================================================
echo  SmartBus TN - Starting EXIT Counter (Camera %CAM%)
echo ============================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment 'venv' not found. Please create it first.
    pause
    exit /b 1
)

python counter.py --door exit --camera %CAM%

echo.
echo ============================================================================
echo  Exit counter session closed.
echo ============================================================================
pause
