from utils.tool_generation import ToolProcessingPipeline

print("=" * 50)
print("[START] Starting process (auto find BaseTool classes + OOP encapsulation)")
print("=" * 50)

try:
    # 加载API配置
    from config.agent_config import openai_api_key, base_url, model_name

    # 校验API密钥
    if not openai_api_key or openai_api_key == "your-openai-api-key":
        raise EnvironmentError("Please set a valid OPENAI_API_KEY in config/agent_config.py")

    # 初始化并运行流水线
    pipeline = ToolProcessingPipeline(openai_api_key, base_url, model_name)
    
    pipeline.run_pipeline()

    # pipeline.release()

except ImportError:
    print("[ERROR] Could not import config from config.agent_config - please check the file exists")
except Exception as e:
    print(f"\n[FATAL ERROR] Program terminated: {str(e)}")
    