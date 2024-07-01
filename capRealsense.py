import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 创建保存视频的目录
output_dir = 'realsense_data'
os.makedirs(output_dir, exist_ok=True)

# 配置RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

# 启动数据流
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_writer = cv2.VideoWriter(os.path.join(output_dir, 'rgb_video.mp4'), fourcc, 30.0, (1920, 1080))
depth_writer = cv2.VideoWriter(os.path.join(output_dir, 'depth_video.mp4'), fourcc, 30.0, (1280, 720), False)
infrared_writer = cv2.VideoWriter(os.path.join(output_dir, 'infrared_video.mp4'), fourcc, 30.0, (1280, 720), False)

# 初始化录制状态
is_recording = False

def update_plot(frame):
    global is_recording

    # 获取相机数据
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    infrared_frame = aligned_frames.get_infrared_frame()

    if not color_frame or not depth_frame or not infrared_frame:
        return

    # 将数据转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    infrared_image = np.asanyarray(infrared_frame.get_data())

    # 更新图像数据
    ax1.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    ax2.imshow(depth_image, cmap='gray')
    ax3.imshow(infrared_image, cmap='gray')

    if is_recording:
        # 在图像上显示红点和"REC"
        ax1.add_patch(patches.Circle((50, 50), radius=20, color='red'))
        ax1.text(80, 55, 'REC', color='red', fontsize=15, fontweight='bold')

        # 写入视频文件
        rgb_writer.write(color_image)
        depth_writer.write(depth_image)
        infrared_writer.write(infrared_image)

def on_key(event):
    global is_recording

    if event.key == ' ':
        is_recording = not is_recording
    elif event.key == 'q':
        plt.close()

# 创建图像窗口
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.canvas.mpl_connect('key_press_event', on_key)

# 开始实时更新
ani = FuncAnimation(fig, update_plot, interval=1)
plt.show()

# 停止数据流
pipeline.stop()
rgb_writer.release()
depth_writer.release()
infrared_writer.release()
cv2.destroyAllWindows()
