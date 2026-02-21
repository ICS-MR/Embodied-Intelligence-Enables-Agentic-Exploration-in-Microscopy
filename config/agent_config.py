from typing import List, Optional

# Prompts (assumed to be imported from local modules)
from prompts.ansis_platform_prompt_full import prompt_imagej
from prompts.cell_seg_prompt_full import prompt_cellpose
from prompts.micro_control_prompt_full import prompt_olympus
from prompts.task_manager_full import prompt_manger
from prompts.fgen import prompt_fgen
from prompts.prompt_check import (
    prompt_no_target,
    prompt_over_exposed,
    prompt_out_of_focus,
    instruction_prompt_without_no_target,
    instruction_prompt_with_no_target
)

Simulation_mode = True  # Set to True for simulation mode, False for real execution

# LLM API Configuration (placeholder values used)
openai_api_key = "YOUR_OPENAI_API_KEY"
base_url = "https://api.chatanywhere.tech"
model_name = "YOUR_LLM_MODEL_NAME"  # e.g., "claude-sonnet-4-5-20250929"

# Vision-Language Model (VLM) API Configuration
vlm_api_key = "YOUR_VLM_API_KEY"
vlm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
vlm_model_name = "YOUR_VLM_MODEL_NAME"  # e.g., "qwen3-vl-235b-a22b-instruct"


# Cross-Encoder Model Path
CROSS_ENCODER_MODEL_PATH = r'model\bge-m3'  # e.g., "./models/bge-m3"
cross_encoder_model_path = CROSS_ENCODER_MODEL_PATH

# Task similarity threshold
task_similarity_threshold = 0.17

# Document paths for RAG
DOCUMENT_PATHS: List[str] = [
    # "RAG_documents/experimental_manual.docx"  # Replace with actual paths
]

# Embedding Configuration
USE_LOCAL_EMBEDDING: bool = True
LOCAL_EMBEDDING_MODEL: str = cross_encoder_model_path
EMBEDDING_API_KEY: Optional[str] = "YOUR_EMBEDDING_API_KEY"
EMBEDDING_BASE_URL: Optional[str] = "YOUR_EMBEDDING_BASE_URL"
EMBEDDING_MODEL: str = "text-embedding-ada-002"

# Text Chunking Settings
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Platform Constraints (in English)
PLATFORM_CONSTRAINTS: str = """
The system supports flexible configuration of multiple imaging parameters, including objectives (4×, 10×, 20×, 40×, 60×), fluorescence channels (e.g., DAPI, FITC, TRITC, Brightfield), light source intensity, and multidimensional acquisition dimensions (Time T, Channel C, Z-stack, XY positions), enabling fully automated 5D image acquisition across T / C / Z / Y / X.

Acquired multi-channel images (compatible with OME-TIFF format) can be automatically processed for channel merging, extended depth-of-field synthesis, deconvolution, adaptive contrast adjustment, and denoising. The system integrates the Cellpose algorithm for high-precision segmentation of nuclei or cytoplasm, and outputs quantitative metrics such as count, area, and morphology.

The system can utilize VLM to perform qualitative analysis of images.

All image acquisition and analysis workflows support extension via Python scripts, allowing users to implement highly customized operations and automated analyses tailored to specific research needs.
"""

# Main Configuration Dictionary
cfg_tabletop = {
    'lmps': {
        'Task_designer': {
            'document_paths': DOCUMENT_PATHS,
            'USE_LOCAL_EMBEDDING': USE_LOCAL_EMBEDDING,
            'LOCAL_EMBEDDING_MODEL': LOCAL_EMBEDDING_MODEL,
            'EMBEDDING_API_KEY': EMBEDDING_API_KEY,
            'EMBEDDING_BASE_URL': EMBEDDING_BASE_URL,
            'EMBEDDING_MODEL': EMBEDDING_MODEL,
            'CHUNK_SIZE': CHUNK_SIZE,
            'CHUNK_OVERLAP': CHUNK_OVERLAP,
            'PLATFORM_CONSTRAINTS': PLATFORM_CONSTRAINTS,
        },
        'Task_manger': {
            'prompt_text': prompt_manger,
            'engine': model_name,
            'max_tokens': 5120,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': '#',
            'maintain_session': True,
            'debug_mode': False,
            'include_context': True,
            'has_return': False,
            'return_val_name': 'ret_val',
        },
        'prompt_olympus': {
            'prompt_text': prompt_olympus,
            'engine': model_name,
            'max_tokens': 5120,
            'temperature': 0,
            'query_prefix': '#',
            'query_suffix': '.',
            'stop': [],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': False,
            'return_val_name': 'ret_val',
        },
        'prompt_imagej': {
            'prompt_text': prompt_imagej,
            'engine': model_name,
            'max_tokens': 5120,
            'temperature': 0,
            'query_prefix': '#',
            'query_suffix': '.',
            'stop': [],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': False,
            'return_val_name': 'ret_val',
        },
        'prompt_cellpose': {
            'prompt_text': prompt_cellpose,
            'engine': model_name,
            'max_tokens': 5120,
            'temperature': 0,
            'query_prefix': '#',
            'query_suffix': '.',
            'stop': [],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': False,
            'return_val_name': 'ret_val',
        },
        'fgen': {
            'prompt_text': prompt_fgen,
            'engine': model_name,
            'max_tokens': 1024,
            'temperature': 0,
            'query_prefix': '# define function: ',
            'query_suffix': '.',
            'stop': [],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
        },
        'checker': {
            'prompt_no_target': prompt_no_target,
            'prompt_over_exposed': prompt_over_exposed,
            'prompt_out_of_focus': prompt_out_of_focus,
            'instruction_prompt_with_no_target': instruction_prompt_with_no_target,
            'instruction_prompt_without_no_target': instruction_prompt_without_no_target,
            'engine': model_name,
            'vlm_engine': vlm_model_name,
            'max_tokens': 1024,
            'temperature': 0,
            'vlm_max_tokens': 1024,
            'vlm_temperature': 0,
            'query_prefix': '# define function: ',
            'query_suffix': '.',
            'stop': [],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
        }
    }
}
