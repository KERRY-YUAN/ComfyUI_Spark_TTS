# Copyright 2025 KERRY-YUAN
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Author: "KERRY-YUAN",
Title: "NodeSparkTTS",
Git-clone: "https://github.com/KERRY-YUAN/ComfyUI_Spark_TTS",
This node package contains nodes for Spark-TTS controllable synthesis and voice cloning.
本节点包旨在实现Spark-TTS的核心功能：可控合成和语音克隆，语音克隆支持预定义说话人和自定义音频输入，并通过简单的逻辑进行选择
"""

import json
import re
import os
import torch
import tempfile
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import platform
import gc
import folder_paths
import torchaudio

current_node_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_node_directory)

from sparktts.utils.file import load_config as spark_load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer as SparkBiCodecTokenizer
from sparktts.utils.token_parser import (
    LEVELS_MAP as SPARK_LEVELS_MAP,
    GENDER_MAP as SPARK_GENDER_MAP,
    TASK_TOKEN_MAP as SPARK_TASK_TOKEN_MAP,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SPARK_MODEL_DIR_GLOBAL: Optional[str] = None
_SPEAKERS_DATA_DIR_GLOBAL: Optional[str] = None
_SPEAKERS_INFO_FILE_GLOBAL: Optional[str] = None

def _get_module_device(module: Optional[torch.nn.Module]) -> Optional[torch.device]:
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            logger.debug(f"Module {type(module).__name__} has no parameters or buffers to determine device.")
            return None
    except Exception as e:
        logger.warning(f"Could not determine device for module {type(module).__name__}: {e}")
        return None

def _initialize_global_paths(): # Removed override parameters
    global _SPARK_MODEL_DIR_GLOBAL, _SPEAKERS_DATA_DIR_GLOBAL, _SPEAKERS_INFO_FILE_GLOBAL
    
    comfyui_root_path_for_defaults = Path(folder_paths.base_path)

    # --- Model Path Handling ---
    default_model_dir = str(comfyui_root_path_for_defaults / "models" / "TTS" / "Spark-TTS" / "Spark-TTS-0.5B")
    if _SPARK_MODEL_DIR_GLOBAL is None or _SPARK_MODEL_DIR_GLOBAL != default_model_dir : # Initialize or update if logic changes
        logger.info(f"Setting Spark-TTS Model Dir to default: {default_model_dir}")
        _SPARK_MODEL_DIR_GLOBAL = default_model_dir
    
    # --- Speaker Preset Path Handling ---
    default_speakers_dir = str(comfyui_root_path_for_defaults / "models" / "TTS" / "Speaker_Preset")
    if _SPEAKERS_DATA_DIR_GLOBAL is None or _SPEAKERS_DATA_DIR_GLOBAL != default_speakers_dir: # Initialize or update
        logger.info(f"Setting Speaker Preset Dir to default: {default_speakers_dir}")
        _SPEAKERS_DATA_DIR_GLOBAL = default_speakers_dir
    
    if _SPEAKERS_DATA_DIR_GLOBAL:
        _SPEAKERS_INFO_FILE_GLOBAL = os.path.join(_SPEAKERS_DATA_DIR_GLOBAL, "speakers_info.json")


if platform.system() == "Darwin" and torch.backends.mps.is_available():
    selected_device = torch.device("mps")
elif torch.cuda.is_available():
    selected_device = torch.device("cuda")
else:
    selected_device = torch.device("cpu")
logger.info(f"SparkTTS determined system device: {selected_device}")

_global_tokenizer: Optional[AutoTokenizer] = None
_global_model: Optional[AutoModelForCausalLM] = None
_global_audio_tokenizer: Optional[SparkBiCodecTokenizer] = None
_current_loaded_model_path: Optional[str] = None

def ensure_models_loaded(device_to_use: torch.device, model_path_for_loading: str):
    global _global_tokenizer, _global_model, _global_audio_tokenizer, _current_loaded_model_path

    if _global_model is not None and _current_loaded_model_path == model_path_for_loading:
        logger.debug("Spark-TTS models already loaded and path matches.")
        llm_device = _get_module_device(_global_model)
        if llm_device is not None and llm_device != device_to_use:
            logger.info(f"Moving existing LLM model from {llm_device} to device: {device_to_use}")
            _global_model.to(device_to_use)

        if _global_audio_tokenizer:
            bicodec_model_instance = getattr(_global_audio_tokenizer, 'model', None)
            bicodec_device = _get_module_device(bicodec_model_instance)
            if bicodec_device is not None and bicodec_device != device_to_use:
                 logger.info(f"Moving existing BiCodec model from {bicodec_device} to device: {device_to_use}")
                 if bicodec_model_instance: bicodec_model_instance.to(device_to_use)
            
            feature_extractor_instance = getattr(_global_audio_tokenizer, 'feature_extractor', None)
            feature_extractor_device = _get_module_device(feature_extractor_instance)
            if feature_extractor_device is not None and feature_extractor_device != device_to_use:
                 logger.info(f"Moving existing Wav2Vec2 feature extractor from {feature_extractor_device} to device: {device_to_use}")
                 if feature_extractor_instance: feature_extractor_instance.to(device_to_use)
        return

    if _global_model is not None: 
        logger.info(f"Spark-TTS model path may have changed or models need reloading. Current loading path: '{model_path_for_loading}', Previously loaded: '{_current_loaded_model_path}'. Reloading...")
        unload_all_models() 

    logger.info(f"Loading Spark-TTS models from: {model_path_for_loading}")
    if not model_path_for_loading or not os.path.exists(model_path_for_loading):
        logger.error(f"Spark-TTS model directory not found: {model_path_for_loading}")
        raise FileNotFoundError(f"Spark-TTS model directory not found: {model_path_for_loading}. Please check path with Model_Download.bat.")
    
    llm_model_subpath = os.path.join(model_path_for_loading, "LLM")
    if not os.path.exists(llm_model_subpath):
        logger.error(f"LLM subdirectory not found in Spark-TTS model directory: {llm_model_subpath}")
        raise FileNotFoundError(f"LLM subdirectory not found in {model_path_for_loading}. Model might be incomplete.")

    try:
        _global_tokenizer = AutoTokenizer.from_pretrained(llm_model_subpath)
        _global_model = AutoModelForCausalLM.from_pretrained(llm_model_subpath)
        _global_model.to(device_to_use).eval()
        _global_audio_tokenizer = SparkBiCodecTokenizer(Path(model_path_for_loading), device=device_to_use)
        _current_loaded_model_path = model_path_for_loading
        logger.info("Spark-TTS models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Spark-TTS models: {e}", exc_info=True)
        unload_all_models()
        raise RuntimeError(f"Failed to load Spark-TTS models from {model_path_for_loading}. Details: {e}") from e

def unload_all_models():
    global _global_tokenizer, _global_model, _global_audio_tokenizer, _current_loaded_model_path
    if _global_tokenizer is not None or _global_model is not None or _global_audio_tokenizer is not None:
        logger.info("Unloading Spark-TTS models.")
    _global_tokenizer = None
    if _global_model is not None:
        del _global_model
        _global_model = None
    if _global_audio_tokenizer is not None:
        if hasattr(_global_audio_tokenizer, 'model'):
            del _global_audio_tokenizer.model
        if hasattr(_global_audio_tokenizer, 'feature_extractor'):
            del _global_audio_tokenizer.feature_extractor
        del _global_audio_tokenizer
        _global_audio_tokenizer = None

    _current_loaded_model_path = None 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _global_model is None and _global_audio_tokenizer is None :
        logger.info("Models and cache cleared successfully.")
    else:
        logger.warning("Models might not have fully unloaded.")

class SparkTTSCoreLogic:
    def __init__(self, device: torch.device, model_base_path_str: str):
        self.device = device
        self.model_base_path = Path(model_base_path_str)
        self.tokenizer: Optional[AutoTokenizer] = _global_tokenizer
        self.model: Optional[AutoModelForCausalLM] = _global_model
        self.audio_tokenizer: Optional[SparkBiCodecTokenizer] = _global_audio_tokenizer
        self.configs: Optional[Dict[str, Any]] = None
        self.sample_rate: int = 16000 

        if self.tokenizer is None or self.model is None or self.audio_tokenizer is None:
            raise RuntimeError("SparkTTSCoreLogic initialized when global models are not available.")

        try:
            config_path = self.model_base_path / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"config.yaml not found in {self.model_base_path}")
            self.configs = spark_load_config(config_path)
            self.sample_rate = self.configs["sample_rate"]
        except Exception as e:
            logger.error(f"Failed to load Spark-TTS config.yaml from {self.model_base_path}: {e}")
            raise RuntimeError(f"Failed to load Spark-TTS model configuration.") from e

    def _prepare_controlled_synthesis_prompt(
        self, text: str, gender: str, pitch_level: str, speed_level: str
    ) -> str:
        if gender not in SPARK_GENDER_MAP:
            raise ValueError(f"Invalid gender: {gender}. Available: {list(SPARK_GENDER_MAP.keys())}")
        if pitch_level not in SPARK_LEVELS_MAP:
            raise ValueError(f"Invalid pitch level: {pitch_level}. Available: {list(SPARK_LEVELS_MAP.keys())}")
        if speed_level not in SPARK_LEVELS_MAP:
            raise ValueError(f"Invalid speed level: {speed_level}. Available: {list(SPARK_LEVELS_MAP.keys())}")

        gender_token = f"<|gender_{SPARK_GENDER_MAP[gender]}|>"
        pitch_token = f"<|pitch_label_{SPARK_LEVELS_MAP[pitch_level]}|>"
        speed_token = f"<|speed_label_{SPARK_LEVELS_MAP[speed_level]}|>"
        attribute_tokens = "".join([gender_token, pitch_token, speed_token])

        prompt_elements = [
            SPARK_TASK_TOKEN_MAP["controllable_tts"], "<|start_content|>", text, "<|end_content|>",
            "<|start_style_label|>", attribute_tokens, "<|end_style_label|>",
        ]
        return "".join(prompt_elements)

    def _prepare_cloning_prompt(
        self, text: str, prompt_speech_path: Path, prompt_text: Optional[str] = None
    ) -> Tuple[str, torch.Tensor]:
        if not prompt_speech_path.exists():
            raise FileNotFoundError(f"Prompt audio file not found: {prompt_speech_path}")

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(str(prompt_speech_path))
        global_tokens_str = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        prompt_elements = [SPARK_TASK_TOKEN_MAP["tts"], "<|start_content|>"]
        if prompt_text and prompt_text.strip():
            semantic_tokens_str = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
            prompt_elements.extend([prompt_text, text, "<|end_content|>"])
        else:
            prompt_elements.extend([text, "<|end_content|>"])
        
        prompt_elements.extend(["<|start_global_token|>", global_tokens_str, "<|end_global_token|>"])
        
        if prompt_text and prompt_text.strip():
            prompt_elements.extend(["<|start_semantic_token|>", semantic_tokens_str])
            
        return "".join(prompt_elements), global_token_ids

    @torch.no_grad()
    def synthesize(
        self, text: str, prompt_speech_path: Optional[Path] = None, prompt_text: Optional[str] = None,
        gender: Optional[str] = None, pitch_level: Optional[str] = None, speed_level: Optional[str] = None,
        temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
        max_new_tokens: int = 6000
    ) -> np.ndarray:
        global_token_ids_for_detokenize: Optional[torch.Tensor] = None
        do_sample_flag = True 

        if gender and pitch_level and speed_level:
            if prompt_speech_path:
                logger.warning("Controllable synthesis mode (gender, pitch, speed provided). Cloning audio prompt will be ignored.")
            logger.info("Performing controllable synthesis.")
            prompt = self._prepare_controlled_synthesis_prompt(text, gender, pitch_level, speed_level)
        elif prompt_speech_path:
            if gender or pitch_level or speed_level:
                 logger.warning("Voice cloning mode selected. Gender, pitch, and speed parameters will be ignored for synthesis.")
            logger.info(f"Performing voice cloning with prompt audio: {prompt_speech_path}")
            prompt, global_token_ids_for_detokenize = self._prepare_cloning_prompt(text, prompt_speech_path, prompt_text)
        else:
            raise ValueError("Insufficient parameters for synthesis. Need either (gender, pitch, speed) or a prompt_speech_path.")

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids_tensor = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample_flag,
            top_k=top_k, top_p=top_p, temperature=temperature,
        )
        input_ids_len = model_inputs.input_ids.shape[1]
        output_ids_only = generated_ids_tensor[:, input_ids_len:]
        
        predicted_tokens_str = self.tokenizer.batch_decode(output_ids_only, skip_special_tokens=False)[0]
        
        semantic_ids_list = [
            int(m.group(1) or m.group(2))
            for m in re.finditer(r"<\|bicodec_semantic_(\d+)\|>|bicodec_semantic_(\d+)", predicted_tokens_str)
        ]
        
        if not semantic_ids_list:
            logger.warning("No semantic tokens (bicodec_semantic_X) found. Output might be empty/incorrect.")
            return np.array([], dtype=np.float32)
        pred_semantic_ids = torch.tensor(semantic_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)

        if gender and global_token_ids_for_detokenize is None: 
            global_ids_list = [
                int(m.group(1) or m.group(2))
                for m in re.finditer(r"<\|bicodec_global_(\d+)\|>|bicodec_global_(\d+)", predicted_tokens_str)
            ]
            if not global_ids_list:
                logger.warning("No global tokens (bicodec_global_X) found in controllable synthesis. This may lead to issues.")
                global_token_ids_for_detokenize = torch.empty((1, 0, 0), dtype=torch.long, device=self.device)
            else:
                 global_token_ids_for_detokenize = torch.tensor(global_ids_list, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)

        if global_token_ids_for_detokenize is None:
            raise RuntimeError("Internal error: Global tokens are missing for detokenization.")

        if global_token_ids_for_detokenize.dim() == 3 and global_token_ids_for_detokenize.shape[0:2] == (1,1) :
            global_tokens_for_detok = global_token_ids_for_detokenize.squeeze(0)
        elif global_token_ids_for_detokenize.dim() == 2 and global_token_ids_for_detokenize.shape[0] == 1:
            global_tokens_for_detok = global_token_ids_for_detokenize
        elif global_token_ids_for_detokenize.numel() == 0:
             global_tokens_for_detok = torch.empty((1,0), dtype=torch.long, device=self.device)
        else:
            global_tokens_for_detok = global_token_ids_for_detokenize.reshape(1, -1)

        wav_output = self.audio_tokenizer.detokenize(
            global_tokens_for_detok.to(self.device),
            pred_semantic_ids.to(self.device),
        )
        logger.info("Speech synthesized successfully.")
        return wav_output

class Spark_TTS_Creation:
    def __init__(self):
        self.tts_core_instance: Optional[SparkTTSCoreLogic] = None

    @classmethod
    def INPUT_TYPES(cls):
        _initialize_global_paths() # Ensures paths are set based on defaults
        return {
            "required": {
                "text": ("STRING", {"default": "Hello, Spark Text to Speech is working!", "multiline": True}),
                # "model_path_override" removed from UI
                "gender": (list(SPARK_GENDER_MAP.keys()), {"default": "female"}),
                "pitch": (list(SPARK_LEVELS_MAP.keys()), {"default": "moderate"}),
                "speed": (list(SPARK_LEVELS_MAP.keys()), {"default": "moderate"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 6000, "min": 100, "max": 90000, "step": 64}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "Keep Model Loaded", "label_off": "Unload Model After Use"}),
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("Audio", "Node Status",)
    FUNCTION = "generate_speech"
    CATEGORY = "ComfyUI_Spark_TTS"

    def generate_speech(self, text: str, gender: str, pitch: str, speed: str,
                        temperature: float, top_k: int, top_p: float, max_new_tokens: int,
                        keep_model_loaded: bool):
        
        _initialize_global_paths() # No override string passed
        
        node_status = "Initializing..."
        final_sample_rate = 16000 
        
        current_model_path_to_load = _SPARK_MODEL_DIR_GLOBAL
        if not current_model_path_to_load or not os.path.isdir(current_model_path_to_load):
            error_msg = f"Model path is invalid or not configured: '{current_model_path_to_load}'. Please run Model_Download.bat or check paths."
            logger.error(error_msg)
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": final_sample_rate}, f"Error: {error_msg}")

        try:
            node_status = f"Loading model from {current_model_path_to_load}..."
            ensure_models_loaded(selected_device, current_model_path_to_load)
            
            node_status = "Creating TTS instance..."
            self.tts_core_instance = SparkTTSCoreLogic(device=selected_device, model_base_path_str=current_model_path_to_load)
            final_sample_rate = self.tts_core_instance.sample_rate

            node_status = "Synthesizing..."
            wav_array = self.tts_core_instance.synthesize(
                text=text, gender=gender, pitch_level=pitch, speed_level=speed,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_tokens=max_new_tokens
            )
            if wav_array.size == 0:
                node_status = "Error: Synthesis resulted in empty audio (no semantic tokens found)."
                logger.error(node_status)
                error_audio_tensor = torch.zeros((1, 1, int(final_sample_rate * 0.1)), dtype=torch.float32)
                return ({"waveform": error_audio_tensor, "sample_rate": final_sample_rate}, node_status)
            
            node_status = "Success"
        except Exception as e:
            error_detail = str(e)
            logger.error(f"Error during Spark_TTS_Creation synthesis: {error_detail}", exc_info=True)
            node_status = f"Error: {error_detail[:200]}"
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": final_sample_rate}, node_status)
        finally:
            if not keep_model_loaded:
                node_status += " Unloading model."
                unload_all_models()
            self.tts_core_instance = None 

        audio_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": final_sample_rate}, node_status)

class Spark_TTS_Clone:
    def __init__(self):
        self.tts_core_instance: Optional[SparkTTSCoreLogic] = None
        self.available_speakers: Dict[str, str] = {} 

    def _refresh_speaker_list_from_path(self, speakers_dir_path: Optional[str]) -> List[str]:
        speaker_names_list = ["(No speakers directory specified)"]
        current_speakers_info_file = None

        if speakers_dir_path and os.path.isdir(speakers_dir_path):
            current_speakers_info_file = os.path.join(speakers_dir_path, "speakers_info.json")
        
        if current_speakers_info_file and os.path.exists(current_speakers_info_file):
            try:
                with open(current_speakers_info_file, "r", encoding="utf-8") as f:
                    speakers_data = json.load(f)
                self.available_speakers = speakers_data 
                loaded_names = list(speakers_data.keys())
                if loaded_names:
                    speaker_names_list = loaded_names
                else:
                    speaker_names_list = ["(speakers_info.json is empty)"]
            except Exception as e:
                logger.error(f"Could not populate speaker list from {current_speakers_info_file}: {e}", exc_info=True)
                self.available_speakers = {"(Error)": f"Could not load {current_speakers_info_file}"}
                speaker_names_list = [f"(Error loading {os.path.basename(current_speakers_info_file)})"]
        elif speakers_dir_path: 
            speaker_names_list = [f"(speakers_info.json not found in {os.path.basename(speakers_dir_path)})"]
        
        return speaker_names_list
        
    @classmethod
    def INPUT_TYPES(cls):
        _initialize_global_paths() # Ensures paths are set based on defaults
        
        temp_instance = cls()
        speaker_names = temp_instance._refresh_speaker_list_from_path(_SPEAKERS_DATA_DIR_GLOBAL)

        return {
            "required": {
                "text": ("STRING", {"default": "Cloning a voice with Spark TTS is interesting.", "multiline": True}),
                "custom_prompt_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Text for Audio reference (optional )"}),
                # "model_path_override" removed from UI
                # "speakers_path_override" removed from UI
                "speaker_preset": (speaker_names, {"default": speaker_names[0] if speaker_names and not speaker_names[0].startswith("(") else ""}),
                "pitch": (list(SPARK_LEVELS_MAP.keys()), {"default": "moderate"}), 
                "speed": (list(SPARK_LEVELS_MAP.keys()), {"default": "moderate"}), 
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 6000, "min": 100, "max": 90000, "step": 64}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "Keep Model Loaded", "label_off": "Unload Model After Use"}),
            },
            "optional": { 
                "Audio_reference": ("AUDIO",), 
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("Audio", "Node Status",)
    FUNCTION = "clone_voice"
    CATEGORY = "ComfyUI_Spark_TTS"

    def clone_voice(self, text: str, custom_prompt_text: str,
                    speaker_preset: str, pitch: str, speed: str, # Removed path overrides from parameters
                    temperature: float, top_k: int, top_p: float, max_new_tokens: int,
                    keep_model_loaded: bool,
                    Audio_reference: Optional[Dict[str, Any]] = None): 
        
        _initialize_global_paths() # No override strings passed
        self._refresh_speaker_list_from_path(_SPEAKERS_DATA_DIR_GLOBAL) 

        node_status = "Initializing..."
        prompt_audio_path_obj: Optional[Path] = None
        effective_prompt_text: Optional[str] = None
        temp_audio_file_path: Optional[str] = None
        final_sample_rate = 16000

        current_model_path_to_load = _SPARK_MODEL_DIR_GLOBAL
        current_speakers_path = _SPEAKERS_DATA_DIR_GLOBAL
        
        if not current_model_path_to_load or not os.path.isdir(current_model_path_to_load):
            error_msg = f"Model path is invalid or not configured: '{current_model_path_to_load}'. Please run Model_Download.bat or check paths."
            logger.error(error_msg)
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": final_sample_rate}, f"Error: {error_msg}")
        if not current_speakers_path or not os.path.isdir(current_speakers_path):
            logger.warning(f"Speaker Preset path is invalid or not configured: '{current_speakers_path}'. Presets might not work. Please run Model_Download.bat.")


        try:
            node_status = f"Loading model from {current_model_path_to_load}..."
            ensure_models_loaded(selected_device, current_model_path_to_load)
            
            node_status = "Creating TTS instance..."
            self.tts_core_instance = SparkTTSCoreLogic(device=selected_device, model_base_path_str=current_model_path_to_load)
            final_sample_rate = self.tts_core_instance.sample_rate

            if Audio_reference and "waveform" in Audio_reference:
                logger.info("Using custom audio reference.")
                node_status = "Processing custom audio reference... "
                waveform_tensor = Audio_reference["waveform"]
                audio_sample_rate = Audio_reference["sample_rate"]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    temp_audio_file_path = tmp_f.name
                
                wf_to_save = waveform_tensor.squeeze(0) if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] == 1 else waveform_tensor
                if wf_to_save.ndim == 1: wf_to_save = wf_to_save.unsqueeze(0) 
                
                torchaudio.save(temp_audio_file_path, wf_to_save.cpu(), audio_sample_rate)
                prompt_audio_path_obj = Path(temp_audio_file_path)
                effective_prompt_text = custom_prompt_text if custom_prompt_text and custom_prompt_text.strip() else None
            elif speaker_preset and not speaker_preset.startswith("("): 
                logger.info(f"Using preset speaker: {speaker_preset}")
                node_status = f"Using preset speaker: {speaker_preset}... "
                if not current_speakers_path or not os.path.isdir(current_speakers_path): # Should be caught above, but double check
                    raise FileNotFoundError(f"Speaker Preset path '{current_speakers_path}' is invalid. Please run Model_Download.bat or check paths.")

                if speaker_preset not in self.available_speakers:
                    error_msg = f"Preset speaker '{speaker_preset}' not found in speakers_info.json at '{current_speakers_path}'. Available: {list(self.available_speakers.keys())}"
                    raise ValueError(error_msg)
                
                found_audio = False
                for ext in [".wav", ".WAV", ".mp3", ".flac", ".ogg", ".m4a"]: 
                    potential_path = Path(current_speakers_path) / f"{speaker_preset}_prompt{ext}"
                    if potential_path.exists():
                        prompt_audio_path_obj = potential_path
                        found_audio = True
                        break
                if not found_audio:
                    raise FileNotFoundError(f"Prompt audio file for '{speaker_preset}' not found in {current_speakers_path} with common extensions.")
                effective_prompt_text = self.available_speakers[speaker_preset]
            else:
                raise ValueError("No valid speaker preset or custom audio reference provided.")

            node_status += "Synthesizing..."
            wav_array = self.tts_core_instance.synthesize(
                text=text, prompt_speech_path=prompt_audio_path_obj, prompt_text=effective_prompt_text,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_tokens=max_new_tokens
            )
            if wav_array.size == 0:
                node_status += " Error: Synthesis resulted in empty audio (no semantic tokens found)."
                logger.error(node_status)
                error_audio_tensor = torch.zeros((1, 1, int(final_sample_rate * 0.1)), dtype=torch.float32)
                return ({"waveform": error_audio_tensor, "sample_rate": final_sample_rate}, node_status)
            
            node_status = "Success." 
        except Exception as e:
            error_detail = str(e)
            logger.error(f"Error during Spark_TTS_Clone synthesis: {error_detail}", exc_info=True)
            node_status = f"Error: {error_detail[:200]}"
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": final_sample_rate}, node_status)
        finally:
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                try:
                    os.remove(temp_audio_file_path)
                except Exception as e_del:
                    logger.warning(f"Could not delete temp file {temp_audio_file_path}: {e_del}")
            
            if not keep_model_loaded:
                node_status += " Unloading model."
                unload_all_models()
            self.tts_core_instance = None 

        audio_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": final_sample_rate}, node_status)

NODE_CLASS_MAPPINGS = {
    "Spark_TTS_Creation": Spark_TTS_Creation,
    "Spark_TTS_Clone": Spark_TTS_Clone,
}