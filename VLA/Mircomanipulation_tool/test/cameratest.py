import gxipy as gx
from PIL import Image

def main():
    # 创建设备管理器
    device_manager = gx.DeviceManager()
    
    # 枚举设备
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("No device found")
        return
    
    # 打开第一台设备
    sn = dev_info_list[0].get("sn")
    cam = device_manager.open_device_by_sn(sn)
    print(f"Opened device with SN: {sn}")
    
    # 设置图像宽度和高度
    cam.Width.set(640)
    cam.Height.set(480)
    
    # 设置帧率
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam.AcquisitionFrameRate.set(30.0)
    
    # 设置曝光时间
    cam.ExposureMode.set(gx.GxExposureModeEntry.TIMED)
    cam.ExposureTime.set(10.0)
    
    # 打印当前帧率
    current_frame_rate = cam.CurrentAcquisitionFrameRate.get()
    print(f"Current frame rate: {current_frame_rate} fps")
    
    # 开始采集
    cam.stream_on()
    
    # 获取图像
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Failed to get image")
        cam.stream_off()
        cam.close_device()
        return
    
    # 转换为RGB图像
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print("Failed to convert image to RGB")
        cam.stream_off()
        cam.close_device()
        return
    
    # 将图像转换为numpy数组
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
        print("Failed to convert image to numpy array")
        cam.stream_off()
        cam.close_device()
        return
    
    # 使用PIL保存图像
    image = Image.fromarray(numpy_image, 'RGB')
    image.save("captured_image.jpg")
    print("Image saved as captured_image.jpg")
    
    # 停止采集
    cam.stream_off()
    
    # 关闭设备
    cam.close_device()
    print("Device closed")

if __name__ == "__main__":
    main()