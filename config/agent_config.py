from prompts.ansis_platform_prompt_full import prompt_imagej
from prompts.cell_seg_prompt_full import prompt_cellpose
from prompts.micro_control_prompt_full import prompt_olympus
from prompts.task_manager_full import prompt_manger
from prompts.fgen import prompt_fgen
from prompts.prompt_check import prompt_no_target, prompt_over_exposed, prompt_out_of_focus, instruction_prompt_without_no_target, instruction_prompt_with_no_target


openai_api_key = ''
base_url = ''
model_name = ''



vlm_api_key = "" 
vlm_base_url = "" 
vlm_model_name = "" 

cross_encoder_model_path = r''
task_similarity_threshold = 0.17

cfg_tabletop = {
    'lmps': {
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
            'instruction_prompt_with_no_target':instruction_prompt_with_no_target,
            'instruction_prompt_without_no_target':instruction_prompt_without_no_target,
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
