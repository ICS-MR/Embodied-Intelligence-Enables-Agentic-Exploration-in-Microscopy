# LLM-Miscope: An LLM-Powered Autonomous Experimental System for Intelligent Microscopy
LLM-Miscope is an intelligent agent system that deeply integrates the logical reasoning capabilities of **Large Language Models (LLMs)** with microscope hardware control. Through natural language interaction, it achieves a **fully automated closed-loop workflow** covering experimental instruction understanding, task planning, automated image acquisition, and real-time analysis.

This project is more than just a control script—it functions as an AI experimenter equipped with "perception-decision-execution-self-correction" capabilities. It is designed to address the pain points of traditional biological experiments, such as complex microscope operations, cumbersome workflows, and the difficulty of achieving fully automated closed-loop control.

## ✨ Core Features
- **🗣️ Natural Language Interaction**: Supports issuing complex experimental instructions directly in natural language (English/Chinese).
- **🧠 Autonomous Task Planning**: Built-in `TaskManager` automatically decomposes abstract goals into executable sequences including movement, focusing, channel switching, Z-stack scanning, etc.
- **🔄 Closed-Loop Self-Correction**: Real-time result verification (e.g., image blurriness check, target tracking) during execution. If a failure occurs, it automatically generates correction instructions for retries instead of reporting errors blindly.
- **🔬 Multimodal Imaging Support**: Natively supports automatic switching and multi-dimensional acquisition (XY-Z-T-C) for brightfield and multiple fluorescence channels (DAPI, FITC, TRITC).
- **🧩 Integrated Advanced Analysis**: Seamlessly integrates **Fiji (ImageJ)** for image processing and **Cellpose/MMDetection** for high-precision cell segmentation and target recognition, enabling "what you see is what you get" intelligent positioning.

## 🛠️ System Architecture
The system adopts a modular design consisting of three layers, each with clear responsibilities and collaborative operation mechanisms:

### 1. Agent Layer (Core Decision-Making Layer)
- **Task Manager**: Parses natural language instructions based on LLM to complete task decomposition and step scheduling.
- **LMP (Language Model Program)**: Dynamically generates executable Python code to drive underlying tool execution.
- **Checker**: A visual and logical verifier responsible for real-time quality control (e.g., image sharpness validation) and exception handling to ensure workflow closure.

### 2. Tool Platform Layer (Function Implementation Layer)
- **Microscope Platform** (`tool/microscope.py`): Core hardware control module responsible for basic operations such as focusing, exposure adjustment, stage displacement, and Z-stack scanning.
- **Image Analysis Platform** (`tool/fiji.py`): Wrapped based on ImageJ, providing image preprocessing, signal analysis and other functions.
- **Cell Segmentation Platform** (`tool/cellpose_tool.py`): Integrates the Cellpose algorithm to realize biological phenotype analysis such as cell segmentation and counting.

### 3. Hardware Driver Layer (Underlying Adaptation Layer)
Unified control of mainstream microscope hardware such as Olympus based on `pymmcore-plus`, providing standardized hardware interaction interfaces to reduce device adaptation costs.

## 🚀 Quick Deployment Guide
Follow the step-by-step process below to complete system initialization:

### 1. Prerequisites
- **Operating System**: Windows 10/11 (recommended for optimal Micro-Manager driver support).
- **Python**: 3.10 or higher
- **External Software**:
  - [Micro-Manager 2.0](https://micro-manager.org/) (Mandatory, for hardware driving)
  - [Fiji (ImageJ)](https://imagej.net/software/fiji/) (Mandatory, for image processing)
- **Hardware**: NVIDIA GPU + CUDA is recommended (for accelerating Cellpose and MMDetection inference).

### 2. Install Dependencies
It is recommended to create an independent Conda environment to avoid dependency conflicts. Execute the following commands:
```bash
# 1. Create environment
conda create -n miscope python=3.10
conda activate miscope

# 2. Install basic project dependencies
pip install -r requirements.txt

# 3. Install MMDetection related libraries (mim tool is recommended)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### 3. Core Configuration (Critical Steps)
Before running, you must modify the following configuration files to adapt to your hardware paths and environment parameters:

#### A. System Path Configuration (config/system_config.py)
Open the file and modify the following key variables to actual paths:
```python
# 1. Micro-Manager hardware configuration file path (.cfg)
CONFIG_PATH = r"C:\path\to\Your_Microscope_Config.cfg"

# 2. Micro-Manager installation root directory (contains device adapters)
MM_DIR = r"C:\Program Files\Micro-Manager-2.0"

# 3. Fiji (ImageJ) application directory
FIJI_PATH = r"D:\Software\Fiji.app"
```

#### B. Model Weight Configuration (config/system_config.py)
Download relevant model weight files, place them in the weights/ directory, or modify the following paths to their actual storage locations:
```python
# For example:
TUMOR_MODEL_CONFIG = "configs/tumor_model.py"
TUMOR_MODEL_CHECKPOINT = "weights/tumor_best.pth"
# ... Configure paths for organoid, 2Dcell and other models as needed
```

#### C. API Key Configuration (config/agent_config.py)
Configure LLM service parameters to ensure the agent works properly:
```python
openai_api_key = 'sk-...'       # Your LLM API Key
base_url = 'https://...'        # API base URL (e.g., OpenRouter or OpenAI)
model_name = 'gpt-4o'           # Model name (gpt-4o or claude-3-5-sonnet is recommended)
```

#### D. Tool Preparation (config/agent_config.py)
Take the FRAP tool as an example to configure the hardware/software tools required by the system. Follow these steps:

1. Place the tool class (e.g., Frap.py) in the tool/ directory. Example code:
```python
# tool/Frap.py
class Frap:
    """A demo tool class for image processing and motion control (matches the prompt constraints in 1_create_tool.py)."""
    
    def __init__(self, device_id: int = 0, save_dir: str = "./output"):
        """Initialize FRAP device and create save directory if not exists."""
        self.device = cv.VideoCapture(device_id)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def capture_image(self, filename: str, resize: tuple = (640, 480)) -> bool:
        """
        Capture an image from the device and save it to the specified path.
        
        Args:
            filename: Name of the image file (e.g., "frame.jpg")
            resize: Tuple of (width, height) for resizing the image
        Returns:
            True if capture/save succeeds, False otherwise
        """
        ret, frame = self.device.read()
        if not ret:
            return False
        frame = cv.resize(frame, resize)
        save_path = os.path.join(self.save_dir, filename)
        return cv.imwrite(save_path, frame)

    def adjust_exposure(self, exposure_value: int) -> None:
        """
        Adjust the device's exposure parameter (hardware security constraint: verify parameter range).
        
        Args:
            exposure_value: Exposure value (must be between 0 and 100)
        Returns:
            None
        """
        if 0 <= exposure_value <= 100:
            self.device.set(cv.CAP_PROP_EXPOSURE, exposure_value)
        else:
            raise ValueError("Exposure value must be between 0 and 100")

    def close_device(self) -> None:
        """Release the device resource and close the capture stream."""
        self.device.release()
```

2. Open `1_create_tool.py` and modify the parameters in the main program section:
```python
if __name__ == "__main__":
    from config.agent_config import openai_api_key, base_url, model_name
    api_key = openai_api_key
    if not api_key:
        raise EnvironmentError("Please set environment variable: OPENAI_API_KEY")

    llm = LLMAgent(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.2,  # 0.1-0.3 is recommended for document generation
        timeout=15         # API timeout period
    )

    folder = "tool"                           # Directory where tool classes are located
    target_class = "Frap"                     # Target class name
    task_manager_file = 'prompts/task_manager_full.py'  # Task manager file path

    # No need to modify the following (unless special requirements exist)
    class_names = get_all_class_names(folder)
    if target_class not in class_names:
        raise FileNotFoundError(f"Class '{target_class}' not found in folder '{folder}'")

    public_methods = get_class_public_methods(
        folder=folder,
        class_name=target_class,
        llm=llm,
        auto_generate_docstring=True,  # Automatically generate Docstring
        inject_say=True                # Inject say method
    )

    if not public_methods.strip():
        logger.error(f"❌ No public methods found for class '{target_class}'")
        exit(1)

    prompt_file_path = generate_tool_usage_prompt(
        folder=folder,
        target_class=target_class,
        llm=llm,
        public_methods=public_methods
    )

    update_task_manager_with_summary_and_example(
        target_class=target_class,
        public_methods=public_methods,
        llm=llm,
        task_manager_file = task_manager_file
    )
```

3. Run the script to generate tool configurations:
```bash
python 1_create_tool.py
```

4. Open `2_main.py`, modify the parameters in the main program section, and complete tool mapping:
Add the method list definition and instantiation logic of the new tool in the `initialize_system` function:
```python
def initialize_system(history_dir, filename):
    # ... Original code ...
    
    # ---- New: NewTool Method List ----
    newtool_methods = [
        "newtool_initialize",       # Initialization method
        "calculate_statistics",     # Core function method
        "newtool_shutdown"          # Resource release method
    ]
    
    try:
        env_olympus = MicroscopeController(OUTPUT_DIR, storageManager)
        env_imagej = ImageJProcessor(storageManager, OUTPUT_DIR)
        env_cellpose = Cellpose2D(storageManager, OUTPUT_DIR)
        # ---- New: Initialize new tool instance ----
        env_newtool = NewTool(storageManager, OUTPUT_DIR)
    except Exception as e:
        raise Exception(f"Core environment initialization failed: {str(e)}")
    
    # ... Original tool method mapping ...
    
    # ---- New: NewTool Method Mapping ----
    newtool_vars = {k: getattr(env_newtool, k) for k in newtool_methods if hasattr(env_newtool, k)}
    
    # Merge variables（Update variable merging logic）
    variable_vars = {}
    variable_vars.update(olympus_vars)
    variable_vars.update(imagej_vars)
    variable_vars.update(cellpose_vars)
    variable_vars.update(newtool_vars)  # New tool variables
    variable_vars['say'] = say_capture.say
    
    # ... Original LMP initialization code ...
    
    # ---- New: Initialize new tool LMP instance ----
    prompt_newtool = LMP(
        'prompt_newtool', 
        cfg_tabletop['lmps']['prompt_newtool'],  # Need to add in configuration
        lmp_fgen, 
        fixed_vars, 
        variable_vars, 
        llm_client, 
        historymanager
    )
    
    # ---- Update module mapping table ----
    module_map = {
        'Microscope Operation Platform': prompt_olympus,
        'Cell Segmentation Platform': prompt_cellpose,
        'Image Analysis Platform': prompt_imagej,
        'Data Analysis Platform': prompt_newtool  # New tool mapping
    }
    
    # ---- Update return dictionary ----
    return {
        # ... Original return items ...
        'env_newtool': env_newtool,  # New tool instance
        # ... Other return items ...
    }
```

5. Add the LMP prompt configuration of the new tool in cfg_tabletop of config/agent_config.py:
```python
# config/agent_config.py
cfg_tabletop = {
    'lmps': {
        # ... Original configurations ...
        'prompt_newtool': {
            'prompt_path': 'prompts/prompt_newtool.py',  # Prompt file path
            'temperature': 0.2,
            'max_tokens': 2000
        },
        # ... Other configurations ...
    }
}
```

## 📖 Key Specifications and Notes
### 1. Tool Class Interface Specifications
New tools must comply with the following interface specifications to ensure compatibility with the system:
- **Mandatory Implementation**: `__init__` method that accepts `storage_manager` and `output_dir` parameters.
- **Recommended Implementation**: Initialization method (e.g., `xxx_initialize`) and resource release method (e.g., `xxx_shutdown`).
- **Function Methods**: Parameters must have clear type annotations; return values are recommended to be serializable dictionaries/basic types.
- **File Operations**: File reading and writing must be performed through `storage_manager`; direct file system operations are prohibited.

#### Standard Implementation Example
Step 1: Must pass StorageManager during tool initialization
```python
class NewTool:
    def __init__(self, storage_manager: StorageManager, output_dir: str):
        # Save StorageManager instance (naming recommendation: _storagemanger or storage_manager)
        self._storagemanger = storage_manager  
        self.output_dir = output_dir  # Tool output directory (used with StorageManager)
```

Step 2: Must register metadata when generating files
```python
def generate_result_file(self, data: pd.DataFrame, filename: str, desc: str):
    # 1. Write file
    file_path = Path(self.output_dir, filename)
    data.to_csv(file_path, index=False)
    
    # 2. Core: Register to StorageManager
    self._storagemanger.register_file(
        filename=filename,
        description=desc,       # e.g., "NewTool statistical analysis result"
        tool_name='newtool',    # Tool name (consistent with module mapping table)
        file_type='csv'         # File format
    )
    return file_path
```

Step 3: Use StorageManager to manage cache
```python
def run_analysis(self, input_data):
    # Clear cache before execution (avoid interference from old data)
    self._storagemanger.clear_cache()
    
    # Execute core logic and generate temporary files...
    
    # Read file list in cache (for subsequent verification)
    cached_files = self._storagemanger.read_cache()
    if not cached_files:
        raise RuntimeError("No analysis results generated!")
```

Step 4: Clean up cache when releasing resources
```python
def shutdown(self):
    # Clean up temporary cache generated by the tool
    self._storagemanger.clear_cache()
    print("NewTool resources released")
```

## 📋 User Guide
### 1. Start the System
Run the main program in the project root directory. The system will automatically complete hardware initialization, service startup, and model loading:
```bash
python 2_main.py
```
After the console displays "System initialization completed successfully", you can enter natural language instructions for interaction.

## 📂 Project Structure
The project directory structure is clear, with core modules divided as follows:
```plaintext
llm_miscope/
├── 2_main.py               # [Entry] Main program entry, responsible for system initialization and loop interaction
├── requirements.txt        # Project dependency list
├── agent/                  # [Core] Core logic of the agent
│   ├── task_agent.py       # Task understanding and decomposition (TaskManager)
│   ├── tool_agent.py       # Code generation and execution (LMP)
│   ├── check_agent.py      # Result checking and feedback (Checker)
│   └── ...
├── tool/                   # [Tools] Underlying tool library
│   ├── microscope.py       # Microscope control core (MicroscopeController)
│   ├── fiji.py             # ImageJ call wrapper (ImageJProcessor)
│   ├── cellpose_tool.py    # Cellpose wrapper (Cellpose2D)
│   └── tool_utils.py       # General tools such as image sharpness calculation
├── config/                 # [Config] Configuration files
│   ├── system_config.py    # Hardware, path and model configuration
│   ├── agent_config.py     # LLM API parameter configuration
│   └── task_config.py      # Task-related constants
├── prompts/                # [Prompts] Prompt engineering
│   ├── task_manager_full.py
│   ├── micro_control_prompt_full.py
│   └── ...
└── weights/                # [Model] Model weight storage directory (need to be added by users)
```

## ⚠️ Notes and Disclaimer
- **Hardware Safety**: The system involves physical hardware control. Please ensure to set limit parameters such as `Max_Z_position` and `Max_X_position` in `config/system_config.py` according to actual microscope parameters to prevent the objective lens from colliding with the glass slide.
- **Model Files**: The target detection function relies on MMDetection model weights. Before the first run, ensure that relevant weight files have been downloaded and the path configuration is correct; otherwise, the related functions will be unavailable.
- **Driver Compatibility**: Ensure that the installed Micro-Manager version is compatible with your microscope hardware, and the .cfg configuration file has been tested to be correct through the official Micro-Manager software.

## 🤝 Contribution
Welcome to submit Issues or Pull Requests to improve this project! If you encounter problems during use, please check the log files in the output/ directory for troubleshooting.

