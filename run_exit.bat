@echo off
REM ============================================================================
REM SmartBus TN - Exit Counter Launcher
REM Edit SERVER_BASE and camera index (CAM) below as needed.
REM Put your INGEST_KEY in a plain text file called ingest_key.txt (one line,
REM no quotes). Never commit that file to git.
REM ============================================================================
set CAM=1
set SERVER_BASE=http://127.0.0.1:8000

REM Load the ingest key from ingest_key.txt (never hard-coded here)
if exist ingest_key.txt (
    set /p INGEST_KEY=<ingest_key.txt
) else (
    echo [WARNING] ingest_key.txt not found. Events will NOT be sent.
    set INGEST_KEY=
)

echo ============================================================================
echo  SmartBus TN - Starting EXIT Counter (Camera %CAM%)
echo  Server: %SERVER_BASE%
echo ============================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment 'venv' not found.
    pause
    exit /b 1
)

python counter.py --door exit --camera %CAM%

echo.
echo ============================================================================
echo  Exit counter session closed.
echo ============================================================================
pause
