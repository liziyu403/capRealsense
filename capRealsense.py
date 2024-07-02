import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

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

# 初始化录制状态
is_recording = False
recording_index = 1
show_depth = False  # 控制是否展示深度图像
rgb_writer = None
depth_writer = None
infrared_writer = None

def start_recording():
    global rgb_writer, depth_writer, infrared_writer, recording_index
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rgb_writer = cv2.VideoWriter(os.path.join(output_dir, f'rgb_video_{recording_index}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))
    depth_writer = cv2.VideoWriter(os.path.join(output_dir, f'depth_video_{recording_index}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720), False)
    infrared_writer = cv2.VideoWriter(os.path.join(output_dir, f'infrared_video_{recording_index}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720), False)

def stop_recording():
    global rgb_writer, depth_writer, infrared_writer, recording_index
    rgb_writer.release()
    depth_writer.release()
    infrared_writer.release()
    recording_index += 1

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

    # 缩小显示分辨率
    display_image = cv2.resize(color_image, (640, 360))

    # 将深度图像归一化到0-255范围，并转换为8位灰度图像
    depth_image_normalized = (depth_image / 65536 * 255).astype(np.uint8)

    if show_depth:
        # 显示RGB和深度图像
        ax[0].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[1].imshow(depth_image_normalized, cmap='gray')
        ax[1].axis('off')
    else:
        # 只显示RGB图像
        ax.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    if is_recording:
        # 写入视频文件
        rgb_writer.write(color_image)
        depth_writer.write(depth_image_normalized)
        infrared_writer.write(infrared_image)

        # 在图像上显示红点和"REC"
        if show_depth:
            ax[0].add_patch(plt.Circle((50, 50), 20, color='red'))
            ax[0].text(80, 55, 'REC', color='red', fontsize=15, fontweight='bold')
        else:
            ax.add_patch(plt.Circle((50, 50), 20, color='red'))
            ax.text(80, 55, 'REC', color='red', fontsize=15, fontweight='bold')
  
    else:
        # 在图像上显示红点和"REC"
        if show_depth:
            ax[0].add_patch(plt.Circle((50, 50), 20, color='white'))
            ax[0].text(80, 55, 'REC', color='white', fontsize=15, fontweight='bold')
        else:
            ax.add_patch(plt.Circle((50, 50), 20, color='white'))
            ax.text(80, 55, 'REC', color='white', fontsize=15, fontweight='bold')

def on_key(event):
    global is_recording, show_depth

    if event.key == ' ':
        if is_recording:
            stop_recording()
            is_recording = False
        else:
            start_recording()
            is_recording = True
    elif event.key == 'q':
        plt.close()
    elif event.key == 'd':
        show_depth = not show_depth

# 创建图像窗口
if show_depth:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
else:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

fig.canvas.mpl_connect('key_press_event', on_key)

# 开始实时更新
ani = FuncAnimation(fig, update_plot, interval=1)
plt.show()

# 停止数据流
pipeline.stop()
if rgb_writer is not None:
    rgb_writer.release()
if depth_writer is not None:
    depth_writer.release()
if infrared_writer is not None:
    infrared_writer.release()
cv2.destroyAllWindows()
