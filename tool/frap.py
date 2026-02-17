from tool.base import BaseTool

class Frap(BaseTool):
    """示例工具类：Frap（继承BaseTool，使用装饰器注册工具函数）"""

    @BaseTool.tool_func  # 装饰器标记：该函数为工具函数，会被注册和提取
    def capture_image(self, resolution: tuple = (1920, 1080), save_path: str = "frap_img.jpg") -> bool:
        """
        Capture a high-resolution image using the Frap device.
        :param resolution: Image resolution (width, height), default is (1920, 1080)
        :param save_path: Path to save the captured image, default is "frap_img.jpg"
        :return: True if capture and save succeed, False otherwise
        """
        # 模拟功能实现（实际场景替换为真实硬件逻辑）
        print(f"[ACTION] Capturing image with resolution {resolution}, saving to {save_path}")
        return True

    @BaseTool.tool_func
    def adjust_exposure(self, exposure_value: int = 50, auto_exposure: bool = False) -> bool:
        """
        Adjust the exposure parameter of the Frap device.
        :param exposure_value: Manual exposure value (0-100), default is 50
        :param auto_exposure: Whether to enable auto exposure, default is False
        :return: True if adjustment succeeds, False otherwise
        """
        # 模拟功能实现
        print(f"[ACTION] Setting exposure: value={exposure_value}, auto={auto_exposure}")
        return True

    # 非工具函数（无装饰器，不会被注册和提取）
    def _calibrate_internal(self) -> None:
        """私有辅助函数：内部校准（不对外暴露）"""
        print("[INFO] Performing internal calibration...")
