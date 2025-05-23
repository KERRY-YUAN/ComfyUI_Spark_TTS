@echo off
setlocal enabledelayedexpansion

REM --- 配置 ---
set "SPARK_TTS_MODEL_URL=https://huggingface.co/SparkAudio/Spark-TTS-0.5B"
set "SPEAKER_PRESET_REPO_URL=https://github.com/KERRY-YUAN/Speaker_Preset.git"

REM --- 确定脚本所在目录 (ComfyUI_Spark_TTS 节点目录) ---
set "NODE_DIR=%~dp0"
REM 去掉末尾的反斜杠
set "NODE_DIR=%NODE_DIR:~0,-1%"

REM --- 尝试自动检测 ComfyUI 根目录 ---
REM 逻辑：从节点目录向上查找，直到找到包含 "main.py" 或 "models" 文件夹的目录
set "COMFYUI_ROOT_DIR="
set "CURRENT_CHECK_DIR=%NODE_DIR%"

:find_comfyui_root_loop
    REM 检查当前目录是否包含 main.py 或 models 文件夹
    if exist "%CURRENT_CHECK_DIR%\main.py" (
        set "COMFYUI_ROOT_DIR=%CURRENT_CHECK_DIR%"
        goto found_comfyui_root
    )
    if exist "%CURRENT_CHECK_DIR%\models" (
        set "COMFYUI_ROOT_DIR=%CURRENT_CHECK_DIR%"
        goto found_comfyui_root
    )

    REM 如果已经是驱动器根目录 (例如 C:\)，则停止
    if "%CURRENT_CHECK_DIR:~-1%"=="\" if "%CURRENT_CHECK_DIR:~-2%"==":\" (
        goto not_found_comfyui_root_auto
    )
    if "%CURRENT_CHECK_DIR%"=="%CURRENT_CHECK_DIR%\" ( REM 处理类似 X:\ 的情况
         if "%CURRENT_CHECK_DIR:~-2%"==":\" (
            goto not_found_comfyui_root_auto
         )
    )


    REM 获取父目录
    for %%F in ("%CURRENT_CHECK_DIR%\..") do set "PARENT_DIR=%%~fF"

    REM 如果父目录和当前目录相同，说明到达了顶层，停止
    if "%PARENT_DIR%"=="%CURRENT_CHECK_DIR%" (
        goto not_found_comfyui_root_auto
    )

    set "CURRENT_CHECK_DIR=%PARENT_DIR%"
    goto find_comfyui_root_loop

:not_found_comfyui_root_auto
    echo.
    echo WARNING: Could not automatically detect ComfyUI root directory.
    REM Fallback: 假设 ComfyUI 根目录是 custom_nodes 的父目录
    for %%F in ("%NODE_DIR%\..\..") do set "COMFYUI_ROOT_DIR=%%~fF"
    echo Using fallback ComfyUI root: %COMFYUI_ROOT_DIR%
    echo Please verify this is correct. If not, manually set COMFYUI_ROOT_DIR in this script.
    echo.
    goto continue_with_paths

:found_comfyui_root
    echo ComfyUI root directory detected: %COMFYUI_ROOT_DIR%
    echo.

:continue_with_paths
    if not defined COMFYUI_ROOT_DIR (
        echo ERROR: COMFYUI_ROOT_DIR could not be determined. Exiting.
        pause
        exit /b 1
    )

REM --- 目标路径设置 ---
set "MODELS_TTS_DIR=%COMFYUI_ROOT_DIR%\models\TTS"
set "SPARK_MODEL_TARGET_BASE_DIR=%MODELS_TTS_DIR%\Spark-TTS"
set "SPARK_MODEL_TARGET_DIR=%SPARK_MODEL_TARGET_BASE_DIR%\Spark-TTS-0.5B"
set "SPEAKER_PRESET_TARGET_DIR=%MODELS_TTS_DIR%\Speaker_Preset"

REM --- 确保目标父目录存在 ---
if not exist "%MODELS_TTS_DIR%" (
    echo Creating directory: %MODELS_TTS_DIR%
    mkdir "%MODELS_TTS_DIR%"
)
if not exist "%SPARK_MODEL_TARGET_BASE_DIR%" (
    echo Creating directory: %SPARK_MODEL_TARGET_BASE_DIR%
    mkdir "%SPARK_MODEL_TARGET_BASE_DIR%"
)


REM --- 下载 Spark-TTS-0.5B 模型 ---
echo.
echo --- Downloading Spark-TTS-0.5B Model ---
echo Target directory: %SPARK_MODEL_TARGET_DIR%

if exist "%SPARK_MODEL_TARGET_DIR%" (
    REM 检查目录是否为空的简单方法：尝试列出内容并检查错误级别
    dir /b "%SPARK_MODEL_TARGET_DIR%" >nul 2>nul
    if errorlevel 1 (
        echo Directory %SPARK_MODEL_TARGET_DIR% exists but is empty. Proceeding with download.
    ) else (
        echo Spark-TTS-0.5B model directory already exists and is not empty. Skipping download.
        echo To update, please delete the directory and re-run this script, or download manually.
        goto download_speaker_preset
    )
)

echo Downloading Spark-TTS-0.5B model from Hugging Face...
echo This requires git to be installed and in your PATH.
echo Cloning %SPARK_TTS_MODEL_URL% into %SPARK_MODEL_TARGET_DIR% ...

git clone %SPARK_TTS_MODEL_URL% "%SPARK_MODEL_TARGET_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone Spark-TTS-0.5B model.
    echo Please check your internet connection, if git is installed and in PATH, or download manually.
    echo Manual download: %SPARK_TTS_MODEL_URL%
    echo And place the contents into: %SPARK_MODEL_TARGET_DIR%
) else (
    echo Spark-TTS-0.5B model downloaded successfully.
)


:download_speaker_preset
REM --- 下载 Speaker_Preset 文件 ---
echo.
echo --- Downloading Speaker_Preset Files ---
echo Target directory: %SPEAKER_PRESET_TARGET_DIR%

if exist "%SPEAKER_PRESET_TARGET_DIR%" (
    dir /b "%SPEAKER_PRESET_TARGET_DIR%" >nul 2>nul
    if errorlevel 1 (
        echo Directory %SPEAKER_PRESET_TARGET_DIR% exists but is empty. Proceeding with download.
    ) else (
        echo Speaker_Preset directory already exists and is not empty. Skipping download.
        echo To update, please delete the directory and re-run this script, or go into the directory and run 'git pull'.
        goto end_script
    )
)

echo Downloading Speaker_Preset files from GitHub...
echo This requires git to be installed and in your PATH.
echo Cloning %SPEAKER_PRESET_REPO_URL% into %SPEAKER_PRESET_TARGET_DIR% ...

git clone %SPEAKER_PRESET_REPO_URL% "%SPEAKER_PRESET_TARGET_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone Speaker_Preset repository.
    echo Please check your internet connection, if git is installed and in PATH, or download manually.
    echo Manual download: %SPEAKER_PRESET_REPO_URL% (Download as ZIP and extract)
    echo And place the contents into: %SPEAKER_PRESET_TARGET_DIR%
) else (
    echo Speaker_Preset files downloaded successfully.
)


:end_script
echo.
echo --- All download tasks attempted ---
echo Please check the following directories:
echo - Spark-TTS Model: %SPARK_MODEL_TARGET_DIR%
echo - Speaker Presets: %SPEAKER_PRESET_TARGET_DIR%
echo.
pause
endlocal