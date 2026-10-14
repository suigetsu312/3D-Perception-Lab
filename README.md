# KITTI LiDAR Player

支援相機 LiDAR 疊圖、可旋轉平移的 3D 點雲，以及高度上色的 BEV 鳥瞰圖。
啟動時勾選想開的 viewer，可單開或任意組合，所選視窗同步播放。
今天的理論與公式整理在 [視覺化理論](docs/visualization-theory.md)，
3D 操作見 [3D 說明](docs/3d-viewer.md)。

## 執行

```sh
uv sync
uv run python main.py
```

需要可用的桌面顯示環境。資料路徑、預設序列、FPS、點大小與距離色階
直接在 `main.py` 頂端設定。各 viewer 的序列下拉選單列出目前 split
本機已有影格的序列；切換會暫停播放，所有視窗同步載入新序列第一幀。
播放／暫停、進度拖曳、前後逐幀與 FPS
可在視窗操作；2D/BEV 的空白鍵切換播放，左右方向鍵逐幀。
BEV 的 XY/Z 範圍與解析度也在 `main.py` 設定。

## 類別與訊號

- `PlayerWindow`：主 GUI 類，建立控制元件、連接訊號與顯示影像。
- `PlaybackController`：管理位置、播放狀態與 QTimer，組合背景 worker。
- `RenderWorker`：QObject 搬到 QThread，透過 slot 執行投影工作。
- `KittiRenderer`：不依賴 Qt，組合資料序列與相機模型，回傳 BGR 影像。
- `BEVRenderer`：正交投影矩陣、每格最高高度與高度上色。
- `BEVWindow`：重用 2D 播放 GUI，接收 BEV 影像。
- `ViewerSelectionDialog`：啟動時選擇要開的 viewer。

訊號路徑：按鈕／slider → controller slot → `render_requested`
→ worker slot → `frame_ready` → controller → GUI slot。
跨執行緒的訊號由 Qt 佇列送達；只有 GUI 執行緒操作 QWidget 與 QPixmap。

同時只允許一個投影工作。拖曳時保留最新位置，舊結果不更新畫面。
FPS 是目標速度，若投影比設定的間隔慢，播放會變慢並保留每一幀。
2D/3D 顏色代表到 LiDAR 原點的歐氏距離：近紅、遠藍。
BEV 顏色代表每格最高高度：低藍、高紅，預設篩選 -3～3 公尺。

## 測試

```sh
QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v
```

桌面環境下可驗證實際 OpenGL 畫面與三種 viewer 同步：

```sh
uv run python -m tests.check_3d_viewer
VIEWER_CHECK=bev uv run python -m tests.check_viewer_selection
```

`VIEWER_CHECK` 可用 `camera`、`cloud`、`bev` 或逗號分隔的組合。
這只用於自動驗證；正常啟動仍然在視窗裡勾選，資料參數直接改 `main.py`。
