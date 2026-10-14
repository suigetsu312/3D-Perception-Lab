# 3D 點雲視窗

執行 `uv run python main.py` 後勾選要開的 2D、3D、BEV 視窗。
兩者共用 controller：播放、暫停或拖曳任一視窗的進度會同步影格。
關閉其中一個視窗不停止另一個，最後一個視窗關閉時才清理 worker。

## 操作

| 操作 | 效果 |
|---|---|
| 左鍵拖曳 | 繞觀看中心旋轉 |
| 中鍵拖曳，或 Ctrl + 左鍵拖曳 | 平移觀看中心 |
| 滾輪 | 縮放 |
| 重設按鈕（循環箭頭） | 回到以原點為中心的預設視角 |
| 原點按鈕（房屋） | 保留方向與距離，把觀看中心移回原點 |
| XYZ / Grid | 顯示或隱藏座標軸／參考格線 |
| Size | 調整點在螢幕上的大小 |

## 原點在哪裡

白色標記代表感測器座標原點，狀態列也顯示 `LiDAR origin (0, 0, 0)`，
三條軸也從這裡出發。原點不一定是點雲中實際量到的一個點，
因此我們另外建立導航標記；導航標記不做深度遮擋，方便定位。

- 紅色 +X：車輛前方。
- 綠色 +Y：左方。
- 藍色 +Z：上方。
- 單位是公尺，座標軸長 5 公尺，格線間隔 5 公尺。

格線位於 $`z=0`$，也就是感測器高度的水平參考面，**不是路面**。
每一幀的原點代表當時的 LiDAR 位置，不是固定的全球原點。
目前沒有使用 OXTS 位姿補償，所以播放時車子附近的環境會移動。

## 渲染理論

3D 螢幕同樣需要投影：

$$
\mathbf p_{\mathrm{clip}} = \mathbf P\mathbf V\mathbf M\mathbf p_{\mathrm{lidar}}
$$

1. Model（$`\mathbf M`$）：目前原始 LiDAR 座標就是渲染世界座標，$`\mathbf M=\mathbf I_4`$。
2. View（$`\mathbf V`$）：滑鼠旋轉、平移改變虛擬相機相對於點雲的位置與方向。
3. Projection（$`\mathbf P`$）：虛擬相機把 3D 點變成裁切座標，GPU 再除以 $`w`$。
4. 深度測試：同個像素的近點可以遮住遠點。點雲是離散點，空隙仍存在。

這個 $`\mathbf P`$ 是虛擬相機投影，不是 KITTI 的 $`\mathbf P_2`$。$`\mathbf P_2`$ 用於把 LiDAR 投到
實體相機圖片；3D 視窗可以從任意角度看，因此不使用它限制視野。
每幀更新點的位置與顏色，不重設相機，所以旋轉後仍能繼續播放。

## 組合與訊號

- `PointCloudWindow`：GUI 主類，建立控制元件並連接 signals/slots。
- `PointCloudScene`：組合 GLViewWidget，管理點雲、原點、座標軸與相機。
- `KittiRenderer.render_with_cloud`：背景讀檔，同時提供 2D 影像與 3D 資料。
- `PlaybackController.cloud_ready`：只有被接受的最新影格才發給 3D 視窗。

OpenGL 元件只在 GUI thread 更新，worker 不操作 widget。
需要支援 OpenGL 的桌面環境；Qt 的 offscreen 平台通常無法建立
QOpenGLWidget，因此純離屏單元測試不能證明實際 3D 畫面有成功渲染。

套件資料：[GLViewWidget](https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/glviewwidget.html)、
[GLScatterPlotItem](https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/glscatterplotitem.html)。

## 實際畫面驗證

```sh
uv run python -m tests.check_3d_viewer
```

這個檢查使用實際 KITTI 序列與 OpenGL framebuffer，驗證非空點雲、
旋轉、平移、縮放、原點定位、小視窗，以及 2D/3D 同步播放。
預覽圖寫到 `outputs/qt_3d*.png`。測試的資料路徑目前與 main 預設相同。
