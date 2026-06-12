@echo off
REM ============================================================================
REM KataRank API Server — startup script
REM ============================================================================
REM Usage:
REM   scripts\katarank-server.bat --model path\to\kata1.bin.gz [options]
REM
REM Options (same as katarank-server CLI):
REM   --model MODEL      REQUIRED  KataGo model .bin.gz
REM   --checkpoint PATH  Optional   KataRankModel .pt checkpoint
REM   --katago-bin PATH  Optional   KataGo binary (auto-detected if omitted)
REM   --config PATH      Optional   KataGo config .cfg
REM   --human-model PATH Optional   HumanSL model .bin.gz
REM   --host ADDR        Default    127.0.0.1
REM   --port PORT        Default    8765
REM   --sgf-root PATH    Optional   Restrict file access to this directory
REM   --max-concurrency N Default    1
REM   --engine-mode MODE Default    lite  (lite|full)
REM   --no-persistent    Optional   Spawn fresh katago per request
REM ============================================================================

cd /d "%~dp0.."

REM --- Activate virtual environment ---
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: No .venv found. Run 'uv sync --extra api' first.
    exit /b 1
)

REM --- Start server ---
echo [%DATE% %TIME%] Starting KataRank server...
katarank-server %*
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Server exited with code %ERRORLEVEL%
    pause
)
