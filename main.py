import os
from pickle import FALSE, TRUE

from networkx import planar_layout
from sympy import false, im
os.environ["PYMMCORE_LOG_TO_FILE"] = "0"
os.environ["BFIO_LOG_TO_FILE"] = "0"
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime
import json

import time
import logging

noisy_loggers = [
    "pymmcore_plus",
    "bfio",
    "openai",
    "httpx",
    "httpcore",
    "urllib3",
]

for name in noisy_loggers:
    logging.getLogger(name).setLevel(logging.WARNING)
from openai import OpenAI

from agent.experiment_planner import ExperimentPlanAgent
from utils.memory_manager import HistoryManager,StorageManager
from agent.experiment_executor import ExperimentExecuteAgent
from agent.experiment_checker import ExperimentCheckAgent
from agent.experiment_designer import ExperimentDesignAgent
from config.agent_config import cfg_tabletop, openai_api_key, base_url, model_name, vlm_api_key, vlm_base_url
from core_tool.tool_utils import SayCapture
from core_tool.cellpose_tool import Cellpose2D
from core_tool.fiji import ImageJProcessor
from core_tool.microscope import MicroscopeController
from config.task_config import OUTPUT_DIR, MAX_RETRY_TIMES, RETRY_INTERVAL, HISTORY_DIR
from config.system_config import CONFIG_PATH, MM_DIR


def summarize_spoken_messages(client, spoken_messages):
    if not spoken_messages:
        return "(No spoken output)"

    messages_text = "\n".join(f"- {msg}" for msg in spoken_messages)
    prompt = f"""Summarize the following robot spoken messages into one or two concise and coherent English sentences. Use third-person perspective and do not add any information beyond what is provided:
{messages_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an objective observer tasked with summarizing the robot's verbal behavior."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def summarize_my_spoken_messages(client, spoken_messages):
    if not spoken_messages:
        return "(No spoken output)"

    messages_text = "\n".join(f"- {msg}" for msg in spoken_messages)
    prompt = f"""Summarize the following spoken messages into one or two concise and coherent English sentences, as if you are the speaker describing your own actions or intentions. Use first-person perspective (e.g., 'I will...', 'I am...') and do not add any information beyond what is provided:
{messages_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are summarizing your own spoken messages from a first-person perspective."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


def summarize_task_execution(client, user_command: str, lmp_steps: list) -> str:
    """
    Use an LLM to summarize the user's command and the execution steps.
    """
    if not lmp_steps:
        step_desc = "(No execution steps)"
    else:
        steps = []
        for step in lmp_steps:
            idx = step.get('subtask_index', '?')
            module = step.get('module', 'Unknown')
            cmd = step.get('command', '').strip()
            steps.append(f"{idx}. [{module}] {cmd}")
        step_desc = "\n".join(steps)

    prompt = f"""You are an intelligent lab assistant. Generate a concise English summary of the task execution based on the following information:

    User's original command:
    "{user_command}"

    Actual execution steps:
    {step_desc}

    Summarize the task's implementation process in one or two sentences using third-person, objective tone, and end with a gentle, guiding question about the next step."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You excel at summarizing experimental workflows with concise and professional language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Task summary failed: {str(e)}]"
    


say_capture = SayCapture()

llm_client = OpenAI(api_key=openai_api_key, base_url=base_url)
vlm_client = OpenAI(api_key=vlm_api_key, base_url=vlm_base_url)


def initialize_system(human_mode):
    historymanager = HistoryManager(HISTORY_DIR)
    storageManager = StorageManager(HISTORY_DIR, OUTPUT_DIR)

    try:
        env_olympus = MicroscopeController(CONFIG_PATH, MM_DIR, OUTPUT_DIR, storageManager)
        env_imagej = ImageJProcessor(storageManager, OUTPUT_DIR)
        env_cellpose = Cellpose2D(storageManager, OUTPUT_DIR)
    except Exception as e:
        raise Exception(f"Core environment initialization failed: {str(e)}")

    fixed_vars = {
        'np': np,
        'cv': cv,
        'datetime': datetime,
        'time': time,
        'csv': csv,
        'json': json,
        'plt': plt,
        'say': say_capture.say
    }

    say_capture = SayCapture()
    
    microscope_methods = env_olympus.get_public_methods()
    olympus_vars = {k: getattr(env_olympus, k) for k in microscope_methods if hasattr(env_olympus, k)}
    imagej_methods = env_imagej.get_public_methods()
    imagej_vars = {k: getattr(env_imagej, k) for k in imagej_methods if hasattr(env_imagej, k)}
    cellpose_methods = env_cellpose.get_public_methods()
    cellpose_vars = {k: getattr(env_cellpose, k) for k in cellpose_methods if hasattr(env_cellpose, k)}

    olympus_tool_agent = ExperimentExecuteAgent('prompt_olympus', cfg_tabletop['lmps']['prompt_olympus'], cfg_tabletop['lmps']['fgen'], fixed_vars, olympus_vars, llm_client, historymanager)

    imagej_tool_agent = ExperimentExecuteAgent('prompt_olympus', cfg_tabletop['lmps']['prompt_olympus'], cfg_tabletop['lmps']['fgen'], fixed_vars, imagej_vars, llm_client, historymanager)

    cellpose_tool_agent = ExperimentExecuteAgent('prompt_olympus', cfg_tabletop['lmps']['prompt_olympus'], cfg_tabletop['lmps']['fgen'], fixed_vars, cellpose_vars, llm_client, historymanager)

    plan_agent = ExperimentPlanAgent(
            'Task_manger',
            cfg_tabletop['lmps']['Task_manger'],
            llm_client,
            historymanager,
            clarify_tag=True
        )
    designer = None
    if not human_mode:
        designer = ExperimentDesignAgent(cfg_tabletop['lmps']['Task_designer'], openai_api_key, base_url, model_name)

    check_agent = ExperimentCheckAgent(cfg_tabletop['lmps']['checker'], llm_client, vlm_client, OUTPUT_DIR)

    module_map = {
        'Microscope Operation Platform': olympus_tool_agent,
        'Cell Segmentation Platform': cellpose_tool_agent,
        'Image Analysis Platform': imagej_tool_agent
    }

    storageManager.clear_all_records()
    historymanager.clear()

    return {
        'env_olympus': env_olympus,
        'env_imagej': env_imagej,
        'env_cellpose': env_cellpose,
        'storageManager': storageManager,
        'historymanager': historymanager,
        'task_manager': plan_agent,
        'task_designer': designer,
        'module_map': module_map,
        'checker': check_agent 
    }


def setup_microscope(env_olympus):
    """Set microscope initial state (optimization: add exception handling + status log output)"""
    try:
        env_olympus.initialize()
        env_olympus.set_objective('2-SOB')
        env_olympus.set_channel('1-NONE')
        env_olympus.set_exposure(10)
        env_olympus.set_brightness(100)
        env_olympus.set_z_position(3500)
        env_olympus.set_x_y_position(50000, 50000)
        # Output current configuration status for troubleshooting
        env_olympus.start_preview()
    except Exception as e:
        raise Exception(f"Microscope initial state configuration failed: {str(e)}")

def microscope_info(env_olympus):
        current_state = {
            'objective': env_olympus.get_objective(),
            'channel': env_olympus.get_channel(),
            'exposure': env_olympus.get_exposure(),
            'brightness': env_olympus.get_brightness()
            # 'z_position': env_olympus.get_z_position(),
            # 'xy_position': env_olympus.get_x_y_position()
        }
        return current_state

def process_instruction(system_components, command, human_mode = True):
    """Task instruction parsing (optimization: add logs + exception handling)"""
    try:
        microscope_state = microscope_info(system_components['env_olympus'])
        if not human_mode:
            command = system_components['task_designer'].run(command)
        ready, LMP_steps, tokens = system_components['task_manager'].run(command, microscope_state)
        return ready, LMP_steps, tokens
    except Exception as e:
        raise Exception(f"Instruction parsing failed: {str(e)}")
    

def check_results(storageManager, original_instruction, original_x_y, checker):
    """Result validation (new checker parameter, using instance created during initialization)"""
    try:
        # Get temporary area files
        meta_file_temp = storageManager.read_cache()
        if not meta_file_temp:
            return True, "", False  # No files = validation failed, no correction instruction, no target exception
        if not any(
            info.get("created_by") == "microscope" and info.get("file_type") == "ome-tiff"
            for info in meta_file_temp.values()
        ):
            return True, "", False
        # Batch validate image results (using passed checker instance instead of global)
        all_results = checker.batch_check_from_json(meta_file_temp)
        task_defect_dict = checker.summarize_task_defects()
        print(f"Result validation completed, defect summary: {task_defect_dict}")
        has_no_target_error = False
        all_images_normal = False
        for defect_desc in task_defect_dict.values():
            if "All channels are defect-free" in defect_desc:
                continue
            if "No target" in defect_desc:
                has_no_target_error = True
                # Batch delete invalid cache files
                cache_filenames = list(meta_file_temp.keys())
                storageManager.batch_delete_files(
                    filenames=cache_filenames,
                    delete_physical=True,
                    remove_meta=True
                )
                break

        # Generate correction instruction (you have confirmed the return list, no additional processing needed)
        unified_instruction = checker.generate_task_unified_instruction(
            original_x_y,
            original_instruction=original_instruction
        )
        # Determine if all images are normal (no defects = normal)
        def is_image_defect_free(message: str) -> bool:
            return "All channels are defect-free" in message

        all_images_normal = all(is_image_defect_free(msg) for msg in task_defect_dict.values())


        # Clear detector historical results to avoid cumulative interference
        checker.clear_history_results()
        return all_images_normal, unified_instruction, has_no_target_error
    except Exception as e:
        checker.clear_history_results()
        raise Exception(f"Result validation failed: {str(e)}")

def run_task(
        LMP_steps,
        module_map,
        env_olympus,
        storageManager,
):
    try:
        # Clear cache for this execution (context controlled by caller)
        storageManager.clear_cache()

        for step in sorted(LMP_steps, key=lambda x: x['subtask_index']):
            say_capture.clear()
            meta_file = storageManager.read_log(True)
            context = f"# Saved documents:\n {meta_file}"
            module_name = step['module']
            command = step['command']

            if module_name == 'Microscope Operation Platform':
                env_info = (
                    f"Current xy_position:{env_olympus.get_x_y_position()}, "
                    f"z_position:{env_olympus.get_z_position()}, "
                    f"exposure_time:{env_olympus.get_exposure()}, "
                    f"objective:{env_olympus.get_objective()}, "
                    f"dichroic:{env_olympus.get_channel()}, "
                    f"brightness:{env_olympus.get_brightness()}"
                )
                context += f'\n# Current environment:{env_info}'

            if module_name in module_map:
                module_instance = module_map[module_name]
                module_instance.run(command, context)
            else:
                raise ValueError(f"Unknown module: {module_name}")
            spoken_messages = say_capture.get_messages()
            summary = summarize_spoken_messages(llm_client, spoken_messages)
            print(f'[Robot]{summary}')
        return True  # Execution completed

    except Exception as e:
        print(f"❌ Error occurred during task execution: {str(e)}")
        raise  

def run_task_with_validation(
    original_LMP_steps,
    module_map,
    env_olympus,
    storageManager,
    checker,
    max_retry_times=MAX_RETRY_TIMES,
    retry_interval=RETRY_INTERVAL
):
    """
    High-level orchestration function: Responsible for the complete process of "execute task -> validate results -> decide retry/correction".
    """
    retry_count = 0
    current_steps = [step.copy() for step in original_LMP_steps]

    while retry_count < max_retry_times:
        retry_count += 1

        try:
            # 1. Execute task (call pure execution function)
            original_x_y = env_olympus.get_x_y_position()
            run_task(
                LMP_steps=current_steps,
                module_map=module_map,
                env_olympus=env_olympus,
                storageManager=storageManager
            )
        except Exception as e:
            if retry_count >= max_retry_times:
                return False, retry_count
            time.sleep(retry_interval)
            current_steps = [step.copy() for step in original_LMP_steps]
            continue

        # 2. Validate results
        all_images_normal, unified_instruction, has_no_target_error = check_results(
            storageManager, original_LMP_steps, original_x_y, checker
        )
        print(summarize_my_spoken_messages(llm_client, unified_instruction))
        if all_images_normal:
            storageManager.commit_cache()
            return True, retry_count

        # 3. Decide whether to retry
        if retry_count >= max_retry_times:
            return False, retry_count

        # 4. Prepare next round steps (use correction instruction or fallback to original)
        if unified_instruction and len(unified_instruction) > 0:
            current_steps = [step.copy() for step in unified_instruction]
        else:
            current_steps = [step.copy() for step in original_LMP_steps]

        time.sleep(retry_interval)

    return False, retry_count

def release_resources(system_components):
    """Release system resources"""
    try:
        # Release microscope resources
        env_olympus = system_components.get('env_olympus')
        if env_olympus and hasattr(env_olympus, 'shutdown'):
            env_olympus.shutdown()
        # Release ImageJ resources
        env_imagej = system_components.get('env_imagej')
        if env_imagej and hasattr(env_imagej, 'fiji_shutdown'):
            env_imagej.fiji_shutdown()
        # Clear storage and history
        storageManager = system_components.get('storageManager')
        historymanager = system_components.get('historymanager')
        if storageManager:
            storageManager.clear_cache()
        if historymanager:
            historymanager.clear()
    except Exception as e:
        raise Exception(f"Resource release failed: {str(e)}")



if __name__ == '__main__':

    human_mode = True

    system_components = initialize_system(human_mode)
    try:
        setup_microscope(system_components['env_olympus'])
        while True:
            user_command = input('please\n')
            microscope_state = microscope_info(system_components['env_olympus'])
            ready, LMP_steps, tokens = process_instruction(system_components['task_manager'], user_command, microscope_state)
            if not ready:
                exit(0)

            task_success, retry_times = run_task_with_validation(
                original_LMP_steps=LMP_steps,
                module_map=system_components['module_map'],
                env_olympus=system_components['env_olympus'],
                storageManager=system_components['storageManager'],
                checker=system_components['checker']
            )

            final_summary = summarize_task_execution(
                llm_client,
                user_command=user_command,
                lmp_steps=LMP_steps
            )

            print(final_summary)
    finally:
        release_resources(system_components)

