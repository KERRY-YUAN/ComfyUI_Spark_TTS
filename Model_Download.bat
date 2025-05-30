@echo off
chcp 65001 >nul  REM Set console to UTF-8 / 设置控制台编码为 UTF-8
setlocal enabledelayedexpansion

REM --- Configuration / 配置 ---
set "SPARK_TTS_MODEL_URL=https://huggingface.co/SparkAudio/Spark-TTS-0.5B"
set "SPEAKER_PRESET_REPO_URL=https://github.com/KERRY-YUAN/Speaker_Preset.git"

echo ================================================================================
echo ComfyUI Spark-TTS Model Downloader
echo ComfyUI Spark-TTS 模型下载器
echo ================================================================================
echo.

REM --- Determine script directory (ComfyUI_Spark_TTS node directory) ---
REM --- 确定脚本所在目录 (ComfyUI_Spark_TTS 节点目录) ---
set "NODE_DIR=%~dp0"
set "NODE_DIR=%NODE_DIR:~0,-1%"

REM --- Determine ComfyUI root directory by looking for "main.py" or "models" folder ---
REM --- 确定 ComfyUI 根目录 (通过查找 "main.py" 或 "models" 文件夹) ---
set "COMFYUI_ROOT_DIR="
set "CURRENT_ASCEND_DIR=%NODE_DIR%"
for /L %%i in (1,1,5) do ( REM Ascend up to 5 levels
    if exist "%CURRENT_ASCEND_DIR%\main.py" (
        set "COMFYUI_ROOT_DIR=%CURRENT_ASCEND_DIR%"
        goto found_comfyui_root
    )
    if exist "%CURRENT_ASCEND_DIR%\models" (
        set "COMFYUI_ROOT_DIR=%CURRENT_ASCEND_DIR%"
        goto found_comfyui_root
    )
    for /f "delims=" %%J in ("%CURRENT_ASCEND_DIR%\..") do set "PARENT_DIR_ASCEND=%%~fJ"
    if "%PARENT_DIR_ASCEND%"=="%CURRENT_ASCEND_DIR%" goto :comfyui_root_search_done
    set "CURRENT_ASCEND_DIR=%PARENT_DIR_ASCEND%"
)
:comfyui_root_search_done

:found_comfyui_root
if not defined COMFYUI_ROOT_DIR (
    echo WARNING: ComfyUI root directory not found automatically.
    echo (警告: 未能自动找到 ComfyUI 根目录。)
    REM Fallback to assume ComfyUI root is 2 levels up from NODE_DIR (for standard custom_nodes install)
    for %%F in ("%NODE_DIR%\..\..") do set "COMFYUI_ROOT_DIR=%%~fF"
    echo Using approximate ComfyUI root: %COMFYUI_ROOT_DIR%
    echo (采用近似的 ComfyUI 根目录: %COMFYUI_ROOT_DIR%)
) else (
    echo ComfyUI root directory detected: %COMFYUI_ROOT_DIR%
    echo (检测到的 ComfyUI 根目录: %COMFYUI_ROOT_DIR%)
)
echo.

REM --- Find Python Executable ---
REM --- 查找 Python 可执行文件 ---
set "PYTHON_EXE="
set "PYTHON_FOUND_METHOD="

echo Searching for Python executable... (正在查找 Python 可执行文件...)

:: Priority 1: .venv/Scripts/python.exe within ComfyUI root
if exist "%COMFYUI_ROOT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT_DIR%\.venv\Scripts\python.exe"
    set "PYTHON_FOUND_METHOD=ComfyUI .venv"
    goto found_python
)

:: Priority 2: python_embeded/python.exe or python/python.exe within ComfyUI root
if exist "%COMFYUI_ROOT_DIR%\python_embeded\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT_DIR%\python_embeded\python.exe"
    set "PYTHON_FOUND_METHOD=ComfyUI embedded"
    goto found_python
)
if exist "%COMFYUI_ROOT_DIR%\python\python.exe" (
    set "PYTHON_EXE=%COMFYUI_ROOT_DIR%\python\python.exe"
    set "PYTHON_FOUND_METHOD=ComfyUI local"
    goto found_python
)

:: Priority 3: python_embeded/python.exe or python/python.exe one level above ComfyUI root
for /f "delims=" %%A in ("%COMFYUI_ROOT_DIR%\..") do set "COMFYUI_PARENT_DIR=%%~fA"
if exist "%COMFYUI_PARENT_DIR%\python_embeded\python.exe" (
    set "PYTHON_EXE=%COMFYUI_PARENT_DIR%\python_embeded\python.exe"
    set "PYTHON_FOUND_METHOD=ComfyUI parent embedded"
    goto found_python
)
if exist "%COMFYUI_PARENT_DIR%\python\python.exe" (
    set "PYTHON_EXE=%COMFYUI_PARENT_DIR%\python\python.exe"
    set "PYTHON_FOUND_METHOD=ComfyUI parent local"
    goto found_python
)

:: Priority 4: System Python (via PATH)
where python.exe >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python.exe"
    set "PYTHON_FOUND_METHOD=System PATH"
    goto found_python
)

:python_not_found
echo ERROR: Python executable not found.
echo (错误: 未找到 Python 可执行文件。)
echo Please ensure Python is installed and accessible, or manually set PYTHON_EXE in this script.
echo (请确保 Python 已安装并可访问，或手动在此脚本中设置 PYTHON_EXE 变量。)
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_EXE% (Method: %PYTHON_FOUND_METHOD%)
echo (找到 Python: %PYTHON_EXE% (查找方式: %PYTHON_FOUND_METHOD%))
echo.

REM --- Target Path Setup / 目标路径设置 ---
set "MODELS_TTS_DIR=%COMFYUI_ROOT_DIR%\models\TTS"
set "SPARK_MODEL_TARGET_BASE_DIR=%MODELS_TTS_DIR%\Spark-TTS"
set "SPARK_MODEL_TARGET_DIR=%SPARK_MODEL_TARGET_BASE_DIR%\Spark-TTS-0.5B"
set "SPEAKER_PRESET_TARGET_DIR=%MODELS_TTS_DIR%\Speaker_Preset"

REM --- Ensure target parent directories exist / 确保目标父目录存在 ---
if not exist "%MODELS_TTS_DIR%" (
    echo Creating directory (创建目录): %MODELS_TTS_DIR%
    mkdir "%MODELS_TTS_DIR%"
)
if not exist "%SPARK_MODEL_TARGET_BASE_DIR%" (
    echo Creating directory (创建目录): %SPARK_MODEL_TARGET_BASE_DIR%
    mkdir "%SPARK_MODEL_TARGET_BASE_DIR%"
)


REM --- Run Python Download Script ---
REM --- 运行 Python 下载脚本 ---
set "DOWNLOAD_SCRIPT_PATH=%NODE_DIR%\model_download\model_download.py"

if not exist "%DOWNLOAD_SCRIPT_PATH%" (
    echo ERROR: Download script not found at %DOWNLOAD_SCRIPT_PATH%
    echo (错误: 下载脚本未在以下路径找到: %DOWNLOAD_SCRIPT_PATH%)
    pause
    exit /b 1
)

echo Calling Python download script...
echo (正在调用 Python 下载脚本...)
echo "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT_PATH%"
"%PYTHON_EXE%" "%DOWNLOAD_SCRIPT_PATH%"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Model download script reported an error. See output above.
    echo (错误: 模型下载脚本报告错误。请查看上面的输出。)
    echo If this is due to missing Python packages (e.g., gdown, huggingface_hub, GitPython),
    echo please try installing them manually using: "%PYTHON_EXE%" -m pip install gdown huggingface_hub "gitpython"
    echo (如果错误是由于缺少 Python 包 (例如 gdown, huggingface_hub, GitPython)，)
    echo (请尝试手动安装它们: "%PYTHON_EXE%" -m pip install gdown huggingface_hub "gitpython")
) else (
    echo.
    echo Model download script finished.
    echo (模型下载脚本已完成。)
)


:end_script
echo.
echo --- All download tasks attempted ---
echo --- 所有下载任务已尝试 ---
echo Please check the following directories to ensure files are complete:
echo (请检查以下目录以确保文件完整):
echo - Spark-TTS Model (模型): %SPARK_MODEL_TARGET_DIR%
echo - Speaker Presets (说话人预设): %SPEAKER_PRESET_TARGET_DIR%
echo.
echo Press any key to exit.
echo (按任意键退出。)
pause >nul
endlocal