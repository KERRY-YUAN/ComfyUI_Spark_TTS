import os
import sys
import json
import subprocess
from pathlib import Path
import shutil
import re
import importlib # For checking package existence

# --- Global Variables ---
# Attempt to find the ComfyUI root directory when the script is imported
# This helps in resolving paths correctly if this script is called from NodeSparkTTS.py
COMFYUI_ROOT_DIR_CACHE = None

def get_comfyui_root_dir():
    global COMFYUI_ROOT_DIR_CACHE
    if COMFYUI_ROOT_DIR_CACHE is not None:
        return COMFYUI_ROOT_DIR_CACHE

    try:
        # Try ComfyUI's folder_paths first if available (when called from node)
        import folder_paths
        if hasattr(folder_paths, 'base_path'):
            COMFYUI_ROOT_DIR_CACHE = Path(folder_paths.base_path)
            return COMFYUI_ROOT_DIR_CACHE
    except ImportError:
        pass # folder_paths not available, likely running as standalone script

    # Fallback: ascend from this script's location
    current_dir = Path(__file__).resolve().parent # model_download directory
    # Ascend two levels to get to ComfyUI_Spark_TTS, then two more for ComfyUI root
    # model_download -> ComfyUI_Spark_TTS -> custom_nodes -> ComfyUI
    # This logic might need adjustment based on where ComfyUI is relative to custom_nodes
    # A more robust way is to search for a known file/folder in ComfyUI root
    
    # Search for ComfyUI root by looking for "main.py" or "models" folder
    # Start from parent of "model_download" which is the node's root dir
    search_dir = current_dir.parent 
    while search_dir != search_dir.parent: # Stop at filesystem root
        if (search_dir / "main.py").exists() or \
           (search_dir / "ComfyUI.exe").exists() or \
           (search_dir / "models").is_dir() and (search_dir / "custom_nodes").is_dir(): # More specific check
            COMFYUI_ROOT_DIR_CACHE = search_dir
            return COMFYUI_ROOT_DIR_CACHE
        # If current directory is custom_nodes, then parent is ComfyUI root
        if search_dir.name == "custom_nodes":
            COMFYUI_ROOT_DIR_CACHE = search_dir.parent
            return COMFYUI_ROOT_DIR_CACHE
        search_dir = search_dir.parent
    
    # Last resort fallback, less reliable
    COMFYUI_ROOT_DIR_CACHE = current_dir.parent.parent.parent # ../../../
    print(f"[Model Downloader] Warning: ComfyUI root auto-detection is approximate. Using: {COMFYUI_ROOT_DIR_CACHE}")
    return COMFYUI_ROOT_DIR_CACHE

# Determine Python executable
def get_python_exe_path():
    # Prefer sys.executable if it points inside a venv that ComfyUI might be using
    if "VIRTUAL_ENV" in os.environ or ".venv" in sys.executable or "python_embeded" in sys.executable:
        return sys.executable
    
    # Fallback for standalone execution or if ComfyUI uses system Python
    comfy_root = get_comfyui_root_dir()
    if comfy_root:
        # Windows embedded Python
        win_embedded_python = comfy_root / "python_embeded" / "python.exe"
        if win_embedded_python.exists():
            return str(win_embedded_python)
        # Linux/macOS venv
        unix_venv_python = comfy_root / "venv" / "bin" / "python"
        if unix_venv_python.exists():
            return str(unix_venv_python)
    return "python" # Default to system python if others not found

PYTHON_EXE = get_python_exe_path()


def install_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"[Model Downloader] Package '{package_name}' is already installed.")
        return True
    except ImportError:
        print(f"[Model Downloader] Package '{package_name}' not found. Attempting to install...")
        try:
            subprocess.check_call([PYTHON_EXE, "-m", "pip", "install", package_name])
            print(f"[Model Downloader] Successfully installed '{package_name}'.")
            # Try importing again to confirm
            importlib.import_module(import_name)
            return True
        except Exception as e:
            print(f"[Model Downloader] ERROR: Failed to install package '{package_name}'. Error: {e}")
            print(f"Please try to install it manually: {PYTHON_EXE} -m pip install {package_name}")
            return False

# Ensure download-related packages are installed
# We do this at the top level so it's done once when the module is first imported by NodeSparkTTS.py
# or when model_download.py is run directly.
_PACKAGES_CHECKED = False
def ensure_download_packages():
    global _PACKAGES_CHECKED, gdown, snapshot_download, HfFileSystem, git
    if _PACKAGES_CHECKED:
        return True

    print("[Model Downloader] Checking for required download packages...")
    all_installed = True
    if not install_package("gdown"): all_installed = False
    if not install_package("huggingface_hub"): all_installed = False
    if not install_package("GitPython", "git"): all_installed = False # GitPython imports as 'git'

    if all_installed:
        print("[Model Downloader] All download packages seem to be available.")
        import gdown
        from huggingface_hub import snapshot_download, HfFileSystem
        import git
        _PACKAGES_CHECKED = True
        return True
    else:
        message = "[Model Downloader] ERROR: One or more required packages for downloading models could not be installed. Please see messages above."
        print(message)
        # In a node context, we shouldn't sys.exit. Raise an exception that the node can catch.
        raise ImportError(message)


def download_model_from_info(model_info_entry, comfyui_root_path):
    model_name = model_info_entry['Model']
    address = model_info_entry['Address']
    relative_to_path_str = model_info_entry['To'] # This is relative to ComfyUI root

    # Determine if the "Model" entry refers to a file or a directory basename
    # A simple heuristic: if it has a common file extension, assume it's a file.
    # Otherwise, assume it's a directory to be created/cloned into.
    known_file_extensions = ['.pth', '.safetensors', '.ckpt', '.bin', '.onnx', '.pt', '.json', '.yaml', '.txt', '.zip']
    is_target_a_file = any(model_name.lower().endswith(ext) for ext in known_file_extensions)

    # Base directory where content will be placed (e.g., ComfyUI/models/TTS/)
    target_base_dir = comfyui_root_path / Path(relative_to_path_str)

    if is_target_a_file:
        # If "Model" is a filename, the final path is "To" / "Model"
        final_target_path = target_base_dir / model_name
        # Ensure parent directory for the file exists
        os.makedirs(final_target_path.parent, exist_ok=True)
    else:
        # If "Model" is a directory name, the final path is "To" / "Model" (this becomes the new directory)
        final_target_path = target_base_dir / model_name
        # Ensure the target directory itself exists for cloning/snapshot
        os.makedirs(final_target_path, exist_ok=True)


    print(f"[Model Downloader] Processing '{model_name}':")
    print(f"  Source: {address}")
    print(f"  Target Path: {final_target_path}")

    # Check if target already exists and is non-empty
    if final_target_path.exists():
        if final_target_path.is_file() and final_target_path.stat().st_size > 1024 * 100: # Assume files > 100KB are likely valid
            print(f"  Skipping: File '{final_target_path}' already exists and is reasonably sized.")
            return True
        elif final_target_path.is_dir() and any(final_target_path.iterdir()):
            print(f"  Skipping: Directory '{final_target_path}' already exists and is not empty.")
            return True
        else:
            print(f"  Note: Target '{final_target_path}' exists but is empty or very small. Will attempt to (re)download.")
            if final_target_path.is_dir():
                try: shutil.rmtree(final_target_path)
                except Exception as e: print(f"    Warning: Could not remove existing empty dir {final_target_path}: {e}")
            elif final_target_path.is_file():
                try: os.remove(final_target_path)
                except Exception as e: print(f"    Warning: Could not remove existing file {final_target_path}: {e}")
            # Re-create directory if it was a directory and now removed, for cloning into
            if not is_target_a_file:
                 os.makedirs(final_target_path, exist_ok=True)


    try:
        if "drive.google.com" in address:
            print(f"  Attempting Google Drive download for '{model_name}'...")
            match = re.search(r'[=/]d/([a-zA-Z0-9_-]+)', address) or \
                    re.search(r'id=([a-zA-Z0-9_-]+)', address)
            if match:
                file_id = match.group(1)
                gdown.download(id=file_id, output=str(final_target_path), quiet=False, fuzzy=True)
                print(f"  Successfully downloaded '{model_name}' from Google Drive.")
            else:
                print(f"  ERROR: Could not extract Google Drive file ID from URL: {address}")
                return False
        elif "huggingface.co" in address:
            repo_id_with_potential_filename = address.replace("https://huggingface.co/", "")
            
            # Check if address points directly to a file within a repo
            # e.g., "org/repo/blob/main/file.txt" or "org/repo/resolve/main/file.txt"
            # or just "org/repo" for the whole repo
            parts = repo_id_with_potential_filename.split('/')
            is_hf_direct_file_link = len(parts) > 2 and (parts[-2] in ["blob", "resolve"] or Path(parts[-1]).suffix != "")

            if is_hf_direct_file_link and is_target_a_file:
                # Trying to download a single file
                # Adjust repo_id and filename if full path is given
                if parts[-2] in ["blob", "resolve"]: # e.g. org/repo/blob/main/file.txt
                    actual_filename_in_repo = parts[-1]
                    repo_id = "/".join(parts[:2])
                    print(f"  Attempting Hugging Face single file download for '{actual_filename_in_repo}' from repo '{repo_id}'...")
                    # Make sure parent dir exists for the file
                    os.makedirs(final_target_path.parent, exist_ok=True)
                    # snapshot_download can download single files if filename arg is used
                    # However, hf_hub_download is more direct for single files
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo_id=repo_id, filename=actual_filename_in_repo, local_dir=str(final_target_path.parent), local_dir_use_symlinks=False, force_filename=final_target_path.name)

                else: # e.g. org/repo/file.txt (assuming model_name in json is file.txt and To is org/repo)
                    actual_filename_in_repo = model_name # Use model_name from JSON
                    repo_id = "/".join(parts) # This logic needs care, assuming 'To' + 'Model' gives full HF path
                    # This case is tricky; typically snapshot_download is for repos.
                    # If 'Model' in json is 'file.txt' and 'To' is 'models/HFModels/Org/Repo',
                    # and 'Address' is 'Org/Repo/file.txt', then it should work.
                    # The key is how repo_id and filename for hf_hub_download are formed.
                    # For now, let's assume if it's a file, snapshot_download might try to get the whole repo
                    # unless allow_patterns is used.
                    # A safer bet for single files is hf_hub_download.
                    # Let's assume for now 'snapshot_download' is for whole repos if target is a directory.
                    # If address is 'org/repo' and target is a directory 'ComfyUI/models/HFModels/org/repo/MyModelName'
                    # then snapshot_download to 'ComfyUI/models/HFModels/org/repo/MyModelName'
                    print(f"  Attempting Hugging Face repository/snapshot download for '{repo_id_with_potential_filename}'...")
                    snapshot_download(repo_id=repo_id_with_potential_filename, local_dir=str(final_target_path), local_dir_use_symlinks=False, resume_download=True, ignore_patterns=["*.md", "*.txt", ".gitattributes"])

            elif not is_target_a_file: # Target is a directory, download whole repo
                print(f"  Attempting Hugging Face repository download for '{repo_id_with_potential_filename}'...")
                snapshot_download(repo_id=repo_id_with_potential_filename, local_dir=str(final_target_path), local_dir_use_symlinks=False, resume_download=True, ignore_patterns=["*.md", "*.txt", ".gitattributes"])
            else:
                print(f"  ERROR: Mismatch for Hugging Face link. Target '{model_name}' is a file, but address '{address}' seems like a repo, or vice-versa. Please check model_list.json.")
                return False
            print(f"  Successfully processed '{model_name}' from Hugging Face.")

        elif "github.com" in address:
            print(f"  Attempting GitHub repository clone for '{model_name}'...")
            if final_target_path.exists() and any(final_target_path.iterdir()): # If dir exists and is not empty
                 print(f"  Directory {final_target_path} exists and is not empty. Skipping clone for GitHub repo to avoid overwrite. Please pull manually if update needed.")
                 return True # Consider it 'successful' as it exists
            elif final_target_path.exists(): # Empty dir
                shutil.rmtree(final_target_path)
            
            git.Repo.clone_from(address, str(final_target_path), depth=1)
            print(f"  Successfully cloned '{model_name}' from GitHub.")
        else:
            print(f"  ERROR: Unsupported download address type for '{model_name}': {address}")
            return False
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download/process '{model_name}'. Error: {e}")
        if final_target_path.exists(): # Cleanup partial downloads
            try:
                if final_target_path.is_file(): os.remove(final_target_path)
                elif final_target_path.is_dir(): shutil.rmtree(final_target_path)
            except: pass
        return False

def main():
    print("[Model Downloader] Starting model download process...")
    if not ensure_download_packages():
        # If called from NodeSparkTTS, this exception should be caught there.
        # If run standalone, it will just print the error.
        raise SystemExit("[Model Downloader] Critical download packages could not be installed. Aborting.")

    comfyui_root = get_comfyui_root_dir()
    if not comfyui_root:
        print("[Model Downloader] ERROR: Could not determine ComfyUI root directory. Cannot proceed with downloads.")
        raise SystemExit("[Model Downloader] ComfyUI root directory not found.")
    
    print(f"[Model Downloader] ComfyUI Root detected: {comfyui_root}")

    # Path to model_list.json, relative to this script (model_download.py)
    current_script_dir = Path(__file__).resolve().parent
    model_list_path = current_script_dir / "model_list.json"

    if not model_list_path.exists():
        print(f"[Model Downloader] ERROR: model_list.json not found at {model_list_path}")
        raise FileNotFoundError(f"model_list.json not found at {model_list_path}")

    try:
        with open(model_list_path, 'r', encoding='utf-8') as f:
            model_list_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Model Downloader] ERROR: Failed to parse model_list.json: {e}")
        raise ValueError(f"Error parsing model_list.json: {e}")

    if not isinstance(model_list_data, list):
        print("[Model Downloader] ERROR: model_list.json should contain a JSON array.")
        raise TypeError("model_list.json should be a JSON array.")

    all_successful = True
    for item in model_list_data:
        if not all(k in item for k in ['Model', 'Address', 'To']):
            print(f"[Model Downloader] Warning: Skipping malformed entry in model_list.json: {item}")
            all_successful = False
            continue
        if not download_model_from_info(item, comfyui_root):
            all_successful = False
    
    if all_successful:
        print("\n[Model Downloader] All model processing tasks completed successfully!")
    else:
        print("\n[Model Downloader] Some models may not have been downloaded or processed correctly. Please review the logs.")
        # If run standalone, exit with error code. If imported, let caller handle.
        if __name__ == "__main__":
             sys.exit(1)
        else:
            raise Exception("One or more models failed to download.")

if __name__ == "__main__":
    # This allows the script to be run directly, e.g., by the .bat file
    try:
        main()
        print("Exiting model downloader script.")
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        sys.exit(1)