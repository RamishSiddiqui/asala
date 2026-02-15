@echo off
REM Build script for Unmask Verify binaries on Windows

echo Building Unmask Verify binaries...
echo ==================================

REM Build TypeScript/Node.js packages first
echo.
echo Step 1: Building Node.js packages...
call npm run build
if errorlevel 1 goto error

REM Build Windows binary
echo.
echo Step 2: Building Windows binary...
call npm run binary:win
if errorlevel 1 goto error

echo.
echo ==================================
echo Build complete! Binaries created in .\bin\
echo.
echo Binaries:
dir bin\ /b
goto end

:error
echo.
echo Build failed!
exit /b 1

:end
echo.
echo Done!
