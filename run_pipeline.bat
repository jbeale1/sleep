@echo off
setlocal

set PYTHON=C:\Users\beale\AppData\Local\Programs\Python\Python312\python.exe
set DIR=C:\Users\beale\Documents\2026-sleep\20260224
set SCRIPTS=C:\Users\beale\Documents\2026-sleep\sleep

echo ============================================
echo  Sleep Analysis Pipeline
echo ============================================

echo.
echo [1/8] analyze_sleep_breathing.py
"%PYTHON%" "%SCRIPTS%\analyze_sleep_breathing.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [2/8] breath_envelope_csv.py
"%PYTHON%" "%SCRIPTS%\utils\breath_envelope_csv.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [3/8] analyze_ecg.py --no-plot --csv-out
"%PYTHON%" "%SCRIPTS%\ECG\analyze_ecg.py" --no-plot --csv-out --save-summary "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [4/8] detect_periodic_breathing.py
"%PYTHON%" "%SCRIPTS%\detect_periodic_breathing.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [5/8] analyze_position.py
"%PYTHON%" "%SCRIPTS%\utils\analyze_position.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [6/8] positional_sleep_stats.py
"%PYTHON%" "%SCRIPTS%\utils\positional_sleep_stats.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [7/8] generate_sleep_dashboard.py
"%PYTHON%" "%SCRIPTS%\generate_sleep_dashboard.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo [8/8] plot_sleep_overview.py
"%PYTHON%" "%SCRIPTS%\ECG\plot_sleep_overview.py" "%DIR%"
if errorlevel 1 echo *** FAILED ***

echo.
echo ============================================
echo  Done.
echo ============================================
pause
