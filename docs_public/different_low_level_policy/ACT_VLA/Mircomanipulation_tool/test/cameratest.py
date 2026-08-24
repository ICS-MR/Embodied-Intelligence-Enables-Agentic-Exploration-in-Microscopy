import gxipy as gx
from PIL import Image

def main():
    # Create the device manager.
    device_manager = gx.DeviceManager()
    
    # Enumerate devices.
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("No device found")
        return
    
    # Open the first device.
    sn = dev_info_list[0].get("sn")
    cam = device_manager.open_device_by_sn(sn)
    print(f"Opened device with SN: {sn}")
    
    # Set the image width and height.
    cam.Width.set(640)
    cam.Height.set(480)
    
    # Set the frame rate.
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam.AcquisitionFrameRate.set(30.0)
    
    # Set the exposure time.
    cam.ExposureMode.set(gx.GxExposureModeEntry.TIMED)
    cam.ExposureTime.set(10.0)
    
    # Print the current frame rate.
    current_frame_rate = cam.CurrentAcquisitionFrameRate.get()
    print(f"Current frame rate: {current_frame_rate} fps")
    
    # Start acquisition.
    cam.stream_on()
    
    # Acquire an image.
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Failed to get image")
        cam.stream_off()
        cam.close_device()
        return
    
    # Convert to an RGB image.
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print("Failed to convert image to RGB")
        cam.stream_off()
        cam.close_device()
        return
    
    # Convert the image to a NumPy array.
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
        print("Failed to convert image to numpy array")
        cam.stream_off()
        cam.close_device()
        return
    
    # Save the image with PIL.
    image = Image.fromarray(numpy_image, 'RGB')
    image.save("captured_image.jpg")
    print("Image saved as captured_image.jpg")
    
    # Stop acquisition.
    cam.stream_off()
    
    # Close the device.
    cam.close_device()
    print("Device closed")

if __name__ == "__main__":
    main()
