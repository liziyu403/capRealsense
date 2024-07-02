import cv2

# 视频文件路径
video_path = './realsense_data/rgb_video_1.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
else:
    # 获取视频帧率和分辨率
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', width, height)
    
    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 显示当前帧
        cv2.imshow('Video', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放视频流
    cap.release()
    cv2.destroyAllWindows()
