import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 创建保存图像和深度数据的目录
output_dir = 'realsense_data'
os.makedirs(output_dir, exist_ok=True)

# 配置RealSense相机
pipeline = rs.pipeline()
config = rs.config()

# 启用高分辨率RGB流
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
# 启用高分辨率深度流
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# 启用高分辨率红外流
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

# 启动数据流
pipeline.start(config)

# 创建对齐对象，用于对齐深度图到彩色图
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 等待相机数据
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        infrared_frame = aligned_frames.get_infrared_frame()

        if not color_frame or not depth_frame or not infrared_frame:
            continue

        # 将数据转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        infrared_image = np.asanyarray(infrared_frame.get_data())

        # 创建一个图像窗口展示三幅图像
        plt.figure(figsize=(15, 5))

        # 显示彩色图像
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title('Color Image')
        plt.axis('off')

        # 显示深度图像
        plt.subplot(1, 3, 2)
        plt.imshow(depth_image, cmap='gray')
        plt.title('Depth Image')
        plt.axis('off')

        # 显示红外图像
        plt.subplot(1, 3, 3)
        plt.imshow(infrared_image, cmap='gray')
        plt.title('Infrared Image')
        plt.axis('off')

        # 显示所有图像
        plt.show()

        # 保存图像
        cv2.imwrite(os.path.join(output_dir, 'color_image.png'), color_image)
        cv2.imwrite(os.path.join(output_dir, 'aligned_depth_image.png'), depth_image)
        cv2.imwrite(os.path.join(output_dir, 'infrared_image.png'), infrared_image)

        # 使用OpenCV等待按键事件，按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止数据流
    pipeline.stop()
    cv2.destroyAllWindows()
