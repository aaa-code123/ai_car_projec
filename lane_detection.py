import cv2
import numpy as np


def detect_lane(frame):
    h, w = frame.shape[:2]

    # 1. 灰階 + 高斯模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny 邊緣偵測
    edges = cv2.Canny(blur, 30, 100)

    # 3. ROI 梯形遮罩
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0,          h),
        (int(w*0.3), int(h*0.1)),
        (int(w*0.7), int(h*0.1)),
        (w,          h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    masked = cv2.bitwise_and(edges, mask)

    # 4. Hough 直線偵測
    lines = cv2.HoughLinesP(masked, 1, np.pi/180,
                             threshold=20,
                             minLineLength=30,
                             maxLineGap=150)

    # 5. 分左右車道線，計算偏差量
    left_x,  right_x  = [], []
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            cx = (x1 + x2) // 2
            if slope < -0.3 and cx < w // 2:
                left_x.append(cx)
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3 and cx > w // 2:
                right_x.append(cx)
                right_lines.append((x1, y1, x2, y2))

    result = frame.copy()

    for (x1, y1, x2, y2) in left_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)     # 綠 = 左
    for (x1, y1, x2, y2) in right_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 128, 255), 3)   # 橘 = 右

    x_center = w // 2
    cv2.line(result, (x_center, h), (x_center, h//2), (255, 255, 255), 1)  # 車身中心白線

    if left_x and right_x:
        x_lane = int((np.mean(left_x) + np.mean(right_x)) / 2)
        e = x_lane - x_center
        cv2.line(result, (x_lane, h), (x_lane, h//2), (0, 255, 255), 2)    # 車道中心黃線
        status = f"L+R e={e}px"
    elif left_x:
        e = -40
        status = f"L only e={e}"
    elif right_x:
        e = 40
        status = f"R only e={e}"
    else:
        e = 0
        status = "no lane"

    cv2.polylines(result, roi, True, (255, 200, 0), 1)
    cv2.putText(result, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(result, f"L:{len(left_lines)} R:{len(right_lines)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return e, result


if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)

    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 0)

    e, result = detect_lane(frame)
    cv2.imwrite("lane_test.jpg", result)
    print(f"偏差量 e = {e} px")
    cam.stop()
