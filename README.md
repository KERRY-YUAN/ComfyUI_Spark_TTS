
# ComfyUI_Spark_TTS

A custom node package for ComfyUI that integrates the powerful Spark-TTS text-to-speech model. This package provides nodes for controllable speech synthesis and voice cloning, built upon the core logic of the official Spark-TTS library.

---
---

## Nodes

*   **Spark_TTS_Creation**: Controllably generates speech with specific gender, pitch, and speed.
*   **Spark_TTS_Clone**: Synthesizes speech by cloning a voice from a reference audio (custom uploaded or preset).

## Node Descriptions

### 1. Spark_TTS_Creation (Voice Creation)

*   **Function**: Input text, select parameters like gender, pitch, and speed to generate customized speech.
*   **Key Inputs**:
    *   `text`: The text to be converted to speech.
    *   `model_path_override`: (Optional) Specify the path to the `Spark-TTS-0.5B` model folder.
    *   `gender`: Choose "female" or "male".
    *   `pitch`: Select pitch level (from "very_low" to "very_high").
    *   `speed`: Select speed level (from "very_low" to "very_high").
    *   `temperature`, `top_k`, `top_p`, `max_new_tokens`: Adjust the diversity and length of the generated speech.
    *   `keep_model_loaded`: Choose whether to keep the model in memory after generation (True keeps it, faster; False unloads it, saves VRAM).
*   **Outputs**:
    *   `Audio`: The generated audio.
    *   `Node Status`: Displays the running status or error messages.
*   **Usage**: Fill in the text, adjust parameters, and generate speech.

### 2. Spark_TTS_Clone (Voice Cloning)

*   **Function**: Input text and provide a reference audio; the node will attempt to read the text using the timbre of the reference audio.
*   **Key Inputs**:
    *   `text`: The text to be read with the cloned timbre.
    *   `custom_prompt_text`: (Optional) The transcript corresponding to the reference audio, helps improve cloning quality.
    *   `model_path_override`: (Optional) Specify the path to the `Spark-TTS-0.5B` model folder.
    *   `speakers_path_override`: (Optional) Specify the path to the `Spark-TTS-Speakers` folder (containing `speakers_info.json` and preset audios).
    *   `speaker_preset`: Select a speaker from a preset list as the reference voice (ignored if `Audio_reference` is connected).
    *   `Audio_reference`: (Optional) Connect an external audio source as the reference for voice cloning (e.g., output of a "Load Audio" node). **This takes precedence over `speaker_preset`**.
    *   `pitch`, `speed`: (Optional) Adjust the pitch and speed of the output voice, mainly effective when the cloning signal is not strong or for future features.
    *   `temperature`, `top_k`, `top_p`, `max_new_tokens`: Adjust the diversity and length of the generated speech.
    *   `keep_model_loaded`: Choose whether to keep the model in memory after generation.
*   **Outputs**:
    *   `Audio`: The generated cloned speech.
    *   `Node Status`: Displays the running status or error messages.
*   **Usage**: Fill in the text. Either connect an `Audio_reference` (recommended to also provide `custom_prompt_text`) OR select a preset voice from `speaker_preset`. Then run.

![image](https://github.com/KERRY-YUAN/ComfyUI_Spark_TTS/blob/main/Examples/Spark_TTS_Audio_Clone.png)
---
---

## Installation Steps

1.  **Navigate to ComfyUI `custom_nodes` directory:**
    ```bash
    cd path/to/your/ComfyUI/custom_nodes
    ```
2.  **Clone this repository:**
    ```bash
    git clone https://github.com/KERRY-YUAN/ComfyUI_Spark_TTS ComfyUI_Spark_TTS
    cd ComfyUI_Spark_TTS
    ```
3.  **Install Dependencies:**
    Install the required Python libraries using your ComfyUI Python environment.
    ```bash
    # Example for ComfyUI's embedded Python on Windows:
    # path/to/your/ComfyUI/python_embeded/python.exe -m pip install -r requirements.txt
    # 
    # Or for a system-wide/venv Python:
    pip install -r requirements.txt
    ```
    *Note: Ensure `torch` and `torchaudio` versions are compatible with your system and ComfyUI's existing PyTorch installation. The versions listed are from the original Spark-TTS `requirements.txt`.*

## 📥 Model and Data Setup

You need to download the Spark-TTS model and place the speaker data according to the structure recognized by the nodes.

1.  **Download Spark-TTS 0.5B Model:**
    Download the `Spark-TTS-0.5B` model folder from its [Hugging Face page (SparkAudio/Spark-TTS-0.5B)](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) and place it inside `ComfyUI/models/TTS/Spark-TTS/`.
	
2.  **Speaker Preset Files Location:**
    The `Spark-TTS-Speakers` folder, containing `speakers_info.json` and preset prompt audio files, should be located directly within your `ComfyUI_Spark_TTS` custom node directory: `ComfyUI/custom_nodes/ComfyUI_Spark_TTS/Spark-TTS-Speakers/`. You can customize this path using the "speakers_path_override" input in the `Spark_TTS_Clone` node.

3.  **Directory Structure Reference:**
    The expected final file structure is:

    ```
    ComfyUI/
    ├── custom_nodes/
    │   └── ComfyUI_Spark_TTS/
    │       ├── sparktts/              <-- The sparktts Python package copied from the official SparkTTS repo
    │       │   ├── __init__.py        <-- An empty __init__.py file
    │       │   └── ... (other subdirectories and files, each with __init__.py)
    │       ├── NodeSparkTTS.py
    │       ├── __init__.py            <-- __init__.py for the ComfyUI_Spark_TTS custom node package
    │       ├── requirements.txt
    │       ├── LICENSE
    │       ├── README.md
    │       └── Spark-TTS-Speakers/    <-- Place speaker prompt and info files here
    │           ├── speakers_info.json # Crucial: Ensure this file is valid JSON!
    │           ├── SpeakerName1_prompt.wav
    │           └── ...                # Other speaker prompt audio files
    └── models/
        └── TTS/
            └── Spark-TTS/
                └── Spark-TTS-0.5B/    <-- Place the downloaded Spark-TTS model folder here
                    ├── BiCodec/
                    ├── LLM/
                    ├── wav2vec2-large-xlsr-53/
                    └── config.yaml
    ```
    *   The `sparktts` folder within the `ComfyUI_Spark_TTS` node package is copied from the official Spark-TTS repository. All its subdirectories should contain an `__init__.py` file to be recognized as Python packages.
    *   The `Spark-TTS-Speakers` folder is now part of the custom node package itself. You can override this path in the `Spark_TTS_Clone` node if you wish to store speaker data elsewhere.
    *   The `speakers_info.json` file within the `Spark-TTS-Speakers` folder maps speaker names (for the dropdown in the `Spark_TTS_Clone` node) to their corresponding prompt texts, which you can add to or modify. Example:
        ```json
        {
            "Alice": "This is the reference text for Alice's voice.",
            "Bob_enthusiastic": "Hello there! I am Bob, and I sound very excited!"
        }
        ```
    *   Prompt audio files should be named `{SpeakerName}_prompt.{extension}` (e.g., `Alice_prompt.wav`).

## 📄 License

This project is released under the Apache License 2.0. It utilizes code and models that are based on or derived from projects also released under Apache 2.0.

Please refer to the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   Thanks to the developers of the original [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) project for the powerful model and library components used in these nodes.

---
---

# ComfyUI_Spark_TTS (中文)

一个用于 ComfyUI 的自定义节点包，集成了强大的 Spark-TTS 文本转语音模型。此节点包提供用于可控语音合成和语音克隆的节点，基于官方 Spark-TTS 库的核心逻辑实现。

---
---

## 节点列表

*   **Spark_TTS_Creation**: 可控制地生成具有特定性别、音高和语速的语音。
*   **Spark_TTS_Clone**: 通过参考音频（自定义上传或选择预设）克隆声音来合成语音。

## 节点说明

### 1. Spark_TTS_Creation (语音创作)

*   **功能**: 输入文本，选择性别、音高和语速等参数，生成定制化的语音。
*   **主要输入**:
    *   `text`: 要转为语音的文字。
    *   `model_path_override`: （可选）指定 `Spark-TTS-0.5B` 模型文件夹的路径。
    *   `gender`: 选择“female”或“male”。
    *   `pitch`: 选择音高（从“very_low”到“very_high”）。
    *   `speed`: 选择语速（从“very_low”到“very_high”）。
    *   `temperature`, `top_k`, `top_p`, `max_new_tokens`: 调整语音生成的多样性和长度。
    *   `keep_model_loaded`: 选择是否在生成后保留模型在内存中（True则保留，更快；False则卸载，省显存）。
*   **输出**:
    *   `Audio`: 生成的音频。
    *   `Node Status`: 显示运行状态或错误信息。
*   **用法**: 填入文本，调整参数，即可生成语音。

### 2. Spark_TTS_Clone (语音克隆)

*   **功能**: 输入文本，并提供一个参考音频，节点将尝试用参考音频的音色来朗读文本。
*   **主要输入**:
    *   `text`: 要用克隆音色朗读的文字。
    *   `custom_prompt_text`: （可选）参考音频对应的文字稿，有助于提高克隆效果。
    *   `model_path_override`: （可选）指定 `Spark-TTS-0.5B` 模型文件夹的路径。
    *   `speakers_path_override`: （可选）指定 `Spark-TTS-Speakers` 文件夹（内含 `speakers_info.json` 和预设音频）的路径。
    *   `speaker_preset`: 从预设列表中选择一个说话人作为参考音（如果连接了 `Audio_reference`，则此项无效）。
    *   `Audio_reference`: （可选）连接一个外部音频作为声音克隆的参考（如“加载音频”节点的输出）。**此项优先于 `speaker_preset`**。
    *   `pitch`, `speed`: （可选）调整输出语音的音高和语速，主要在克隆信号不强时或为未来功能预留。
    *   `temperature`, `top_k`, `top_p`, `max_new_tokens`: 调整语音生成的多样性和长度。
    *   `keep_model_loaded`: 选择是否在生成后保留模型在内存中。
*   **输出**:
    *   `Audio`: 生成的克隆语音。
    *   `Node Status`: 显示运行状态或错误信息。
*   **用法**: 填入文本。要么连接一个 `Audio_reference`（推荐同时提供 `custom_prompt_text`），要么从 `speaker_preset` 选择一个预设声音。然后运行即可。

![image](https://github.com/KERRY-YUAN/ComfyUI_Spark_TTS/blob/main/Examples/Spark_TTS_Audio_Clone.png)
---
---

## 安装步骤

1.  **导航到 ComfyUI `custom_nodes` 目录：**
    ```bash
    cd path/to/your/ComfyUI/custom_nodes
    ```
2.  **克隆此仓库：**
    ```bash
    git clone https://github.com/KERRY-YUAN/ComfyUI_Spark_TTS ComfyUI_Spark_TTS
    cd ComfyUI_Spark_TTS
    ```
3.  **安装依赖项：**
    使用您的 ComfyUI Python 环境安装所需的 Python 库。
    ```bash
    # Windows上ComfyUI嵌入式Python示例:
    # path/to/your/ComfyUI/python_embeded/python.exe -m pip install -r requirements.txt
    # 
    # 或者对于系统级/虚拟环境Python:
    pip install -r requirements.txt
    ```
    *注意：请确保 `torch` 和 `torchaudio` 版本与您的系统以及 ComfyUI 现有的 PyTorch 安装兼容。所列版本来自原始 Spark-TTS 的 `requirements.txt`。*

## 📥 模型和数据设置

您需要下载 Spark-TTS 模型，并将说话人数据按照节点识别的特定结构放置。

1.  **下载 Spark-TTS 0.5B 模型：**
    从其 [Hugging Face 页面 (SparkAudio/Spark-TTS-0.5B)](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) 下载 `Spark-TTS-0.5B` 模型文件夹，放在`ComfyUI/models/TTS/Spark-TTS/`内。
	
2.  **说话人预设文件位置：**
    包含 `speakers_info.json` 和预设提示音频文件的 `Spark-TTS-Speakers` 文件夹，现在应该直接位于您的 `ComfyUI_Spark_TTS` 自定义节点目录中：`ComfyUI/custom_nodes/ComfyUI_Spark_TTS/Spark-TTS-Speakers/`。您可以在 `Spark_TTS_Clone` 节点中使用 "speakers_path_override" 输入框来指定此路径的自定义位置。

3.  **目录结构参考：**
    预期的最终文件架构如下：

    ```
    ComfyUI/
    ├── custom_nodes/
    │   └── ComfyUI_Spark_TTS/
    │       ├── sparktts/      <-- 从官方 SparkTTS 仓库复制的 sparktts Python包
    │       │   ├── __init__.py  <-- 空的 __init__.py 文件
    │       │   └── ... (其他子目录和文件，每个都带有 __init__.py)
    │       ├── NodeSparkTTS.py
    │       ├── __init__.py      <-- ComfyUI_Spark_TTS 自定义节点包的 __init__.py
    │       ├── requirements.txt
    │       ├── LICENSE
    │       ├── README.md
    │       └── Spark-TTS-Speakers/      <-- 将说话人提示和信息文件放在此处
    │           ├── speakers_info.json   # 至关重要：确保此文件是有效的 JSON 格式！
    │           ├── 说话人名1_prompt.wav
    │           └── ...                  # 其他说话人提示音频文件
    └── models/
        └── TTS/
            └── Spark-TTS/
            │   └── Spark-TTS-0.5B/      <-- 将下载的 Spark-TTS 模型文件夹放在此处
            │       ├── BiCodec/
            │       ├── LLM/
            │       ├── wav2vec2-large-xlsr-53/
            │       └── config.yaml
    ```
    *   `ComfyUI_Spark_TTS` 节点包内的 `sparktts` 文件夹为官方 Spark-TTS 仓库复制。其所有子目录应包含一个 `__init__.py` 文件，以便被识别为 Python 包。
    *   `Spark-TTS-Speakers` 文件夹现在是自定义节点包的一部分。如果您希望将说话人数据存储在其他位置，可以在 `Spark_TTS_Clone` 节点中覆盖此路径。
    *   `Spark-TTS-Speakers` 文件夹内的`speakers_info.json` 文件将说话人名称（用于 `Spark_TTS_Clone` 节点中的下拉列表）映射到其相应的提示文本，可以自行增减。示例：
        ```json
        {
            "爱丽丝": "这是爱丽丝声音的参考文本。",
            "鲍勃_热情": "你好呀！我是鲍勃，我听起来非常兴奋！"
        }
        ```
    *   提示音频文件应命名为 `{说话人名}_prompt.{扩展名}` (例如, `爱丽丝_prompt.wav`)。

## 📄 许可证

本项目根据 Apache License 2.0 发布。它使用了基于或派生自同样根据 Apache 2.0 发布的项目的代码和模型。

详细信息请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

*   感谢原始 [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) 项目的开发者提供了此节点中使用的强大模型和库组件。

---
---