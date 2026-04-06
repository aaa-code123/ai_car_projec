# 🚗 AI 智慧遙控車控制系統

基於 Raspberry Pi + PiCamera2 的即時車道偵測自動循線小車，提供 Web 控制介面，支援手動遙控與自動循線兩種模式。

---

## 功能特色

- **即時影像串流**：透過瀏覽器觀看 PiCamera2 即時畫面，並疊加車道偵測結果
- **自動循線**：使用 Hough 線段偵測 + PID 控制器自動跟隨白色車道線
- **手動遙控**：鍵盤（WASD）或網頁按鈕控制前後左右
- **鏡頭雲台**：方向鍵控制 Pan/Tilt 舵機，調整攝影角度
- **ROI 拖曳編輯**：在瀏覽器上直接拖曳調整感興趣區域（6 頂點多邊形）
- **即時參數調整**：速度、Kp、FPS、Hough 參數等全部可在前端滑桿即時修改，並自動儲存至 `settings.json`

---

## 系統架構

```
├── app.py              # Flask 主程式，整合所有模組與 API 路由
├── lane_detection.py   # 車道偵測核心（HSV 遮罩 + HoughLinesP + PID 誤差計算）
├── motor_control.py    # PCA9685 馬達驅動 + MotorController 封裝
├── pid_controller.py   # PID 控制器與左右輪速度轉換
├── settings.json       # 持久化參數設定檔（自動讀寫）
└── templates/
    └── index.html      # 單頁控制介面（純 HTML/CSS/JS）
```

---

## 硬體需求

| 元件 | 說明 |
|------|------|
| Raspberry Pi 4 / 5 | 主控板 |
| PiCamera2 | 攝影模組 |
| PCA9685 | 16 通道 PWM 控制器（I²C，地址 0x40） |
| 直流馬達 × 4 | 四輪驅動（A/B/C/D，其中 D 使用 GPIO 25/24） |
| Pan-Tilt 舵機 × 2 | 鏡頭雲台（Channel 9=Pan, 10=Tilt） |

---

## 安裝與執行

### 1. 安裝依賴套件

```bash
pip install flask picamera2 opencv-python-headless numpy smbus2 gpiozero psutil
```

### 2. 下載 YOLOv8 模型

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
```

或透過 ultralytics 自動下載：

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. 啟動伺服器

```bash
python app.py
```

預設監聽 `0.0.0.0:5000`，開啟瀏覽器前往 `http://<Pi的IP>:5000` 即可使用。

---

## 控制介面說明

### 鍵盤快捷鍵

| 按鍵 | 功能 |
|------|------|
| `W / A / S / D` | 前進 / 左轉 / 後退 / 右轉 |
| `↑ ↓ ← →` | 鏡頭上下左右 |
| `空白鍵` | 啟動 / 停止自動循線 |

### 自動循線參數

| 參數 | 說明 |
|------|------|
| `base_speed` | 基礎車速（0~100%） |
| `FPS` | 推論幀率（每秒偵測次數） |
| `Kp` | PID 比例增益（方向矯正力度） |
| `correct_skip` | 每 N 幀執行一次馬達修正 |

### 車道偵測參數

| 參數 | 說明 |
|------|------|
| `white_s_min/max` | HSV 飽和度過濾範圍（白色辨識） |
| `white_v_min/max` | HSV 亮度過濾範圍 |
| `hough_threshold` | Hough 累加器投票門檻 |
| `hough_min_len` | Hough 最小線段長度 |
| `hough_max_gap` | Hough 允許最大線段間隙 |
| `seg_min_len` | 幾何過濾最短線段長度 |
| `lane_half_width` | 車道半寬（px，雙線時自動 EMA 學習） |
| `single_target_ratio` | 單線時目標位置比例（0=最左, 1=最右） |
| `single_e_clamp` | 單線誤差 e 上限，防止過度修正 |

---

## 車道偵測原理

```
原始幀
  │
  ▼
HSV 白色遮罩（飽和度 + 亮度過濾）
  │
  ▼
ROI 6 頂點多邊形遮罩（梯形區域）
  │
  ▼
形態學閉合（填補木紋縫隙）
  │
  ▼
HoughLinesP 線段偵測
  │
  ▼
幾何過濾 + 斜率分類（左線 / 右線）
  │
  ▼
計算車道中心偏差量 e（px）
  │
  ▼
PID → 左右輪速差 → 馬達修正
```

偵測線顏色說明：
- 🟢 綠色：左側車道線
- 🟠 橘色：右側車道線
- 🟡 黃色：估算的車道中心
- ⚪ 白色：車身中心

---

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 控制介面主頁 |
| GET | `/video_feed` | MJPEG 影像串流 |
| POST | `/control` | 手動行駛指令（action: forward/backward/left/right/stop） |
| POST | `/auto` | 自動循線開關（action: start/stop） |
| POST | `/settings` | 更新運行參數（JSON） |
| GET/POST | `/roi` | 取得或更新 ROI 頂點（比例值） |
| POST | `/camera` | 控制鏡頭舵機方向 |
| GET | `/status` | 取得目前狀態、速度、CPU/記憶體使用率 |

---

## 設定持久化

所有參數修改後會自動儲存至 `settings.json`，下次啟動時自動還原，無需重新調整。

---

## 注意事項

- 馬達 D 的方向控制使用 GPIO（`gpiozero.LED`），需確認 GPIO 25/24 未被佔用
- 舵機初始角度為 Pan=90°、Tilt=90°，啟動時會自動歸位
- 手動介入（按 WASD）時會自動關閉自動循線模式
- 建議在白色線條對比明顯的環境下使用，並根據現場光線調整 HSV 參數
