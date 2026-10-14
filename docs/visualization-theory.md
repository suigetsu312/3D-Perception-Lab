# 三種 Viewer 的幾何理論

本專案使用 KITTI Tracking：每個序列包含逐幀相機影像、Velodyne
點雲、calibration 與 training 標註。Tracking 的 track ID 用來識別
跨影格的同一個物件，目前三種 viewer 都只顯示量測，尚未畫 tracking 方框。

## 共用的座標與資料

LiDAR 檔案每筆為四個 little-endian float32：`(x, y, z, reflectance)`。
第四項是反射強度，不是齊次座標！幾何運算先取 xyz，再補成 `(x,y,z,1)`。

LiDAR 軸向為 +X 前方、+Y 左方、+Z 上方，單位公尺。
相機軸向為 +X 右方、+Y 下方、+Z 前方。兩者是不同的座標基底。
白色原點標記是 LiDAR 的 `(0,0,0)`，不是路面，也不是固定的全球位置。
目前沒有 OXTS 位姿補償，因此每幀仍使用當幀的感測器座標系。

公式用列向量 `p' = T @ p`；NumPy 以每列儲存一個點，因此批次寫成
`points @ T.T`。矩陣乘法的順序由右往左讀。

## 齊次座標與剛體轉換

旋轉與平移原本是 `p' = R @ p + t`。補上齊次座標後可合成一次乘法：

```text
T = [ R(3×3)  t(3×1) ]
    [ 0 0 0      1   ]

[x', y', z', 1]ᵀ = T @ [x, y, z, 1]ᵀ
```

平移不影響方向向量，所以方向向量的第四項是 0，位置點則是 1。
這兩種不能混用。對應實作為 `geometry.to_homogeneous` 與 `transform_points`。

## 2D：LiDAR 疊到實體相機影像

序列 0000 的參數在 `data_tracking_calib/training/calib/0000.txt`。
整個序列共用 calibration，換序列時要重新讀取。

| 欄位 | 尺寸 | 功能 |
|---|---|---|
| Tr_velo_cam | 3×4 | 外參：LiDAR → 參考相機，前三欄 R、最後一欄 t |
| R_rect | 3×3 | 參考相機 → 校正後相機座標 |
| P2 | 3×4 | 校正後參考座標 → 左彩色相機 image_02 |

先將 R_rect 與 Tr_velo_cam 擴成 4×4，再計算：

```text
q = P2 @ R_rect @ Tr_velo_cam @ [x, y, z, 1]ᵀ
u = q[0] / q[2]
v = q[1] / q[2]
```

這裡 q 不是已經可以畫的像素，必須做透視除法。
簡化成針孔模型時，`u = fx * X/Z + cx`、`v = fy * Y/Z + cy`。
除以深度 Z 造成近大遠小。相機後方、無效數值與影像外的點要排除。

P2 的左側 3×3 是校正後內參 K。序列 0000 的數值為：

```text
K = [721.5377    0       609.5593]
    [   0     721.5377   172.8540]
    [   0        0         1    ]
```

fx、fy 是像素單位的焦距，cx、cy 是主點。但 P2 的第四欄並不為零，
還包含彩色相機相對校正參考座標的平移效果，所以投影不能只用 K。
提供的影像已 rectified，目前不再施加鏡頭畸變。

2D 顏色表示到 LiDAR 原點的歐氏距離 `sqrt(x²+y²+z²)`，不是相機 Z。
遠點先畫、近點後畫只是一個畫點策略，不是完整的相機遮擋處理。
對應實作：`KittiCamera`、`geometry.project_points`、`KittiRenderer`。

## 3D：自由移動的虛擬相機

3D 螢幕最後仍然是 2D 像素，使用的是虛擬相機：

```text
p_clip = P @ V @ M @ p_local
p_ndc = p_clip.xyz / p_clip.w
```

- Model（M）：物件 → 世界。目前直接用 LiDAR xyz，所以 M 是單位矩陣。
- View（V）：世界 → 虛擬相機。旋轉與平移改變觀看方向及中心。
- Projection（P）：虛擬相機 → 裁切座標，包含 FOV、長寬比與裁切面。
- GPU 除以 w 得到 NDC，再映射到螢幕像素。

這裡 P 不是 KITTI 的 P2，因此能從任意方向看整片點雲。
GPU 的深度測試讓同一像素的近點遮住遠點，但離散點之間仍然有空隙。
導航用的原點與 XYZ 軸刻意關閉深度測試，以免被點雲遮住。

逐幀只更新點的位置與顏色，不重設相機。因此可以邊旋轉邊播放。
3D 點大小以像素計算，與公尺大小的幾何物件不同。
操作及架構見 [3D viewer](3d-viewer.md)。

## BEV：XY 正交投影與柵格化

幾何上，BEV 就是忽略 Z、投到 XY 平面：

```text
B = [1 0 0]
    [0 1 0]
[x_bev, y_bev]ᵀ = B @ [x, y, z]ᵀ
```

它是正交投影，不除以深度、不會近大遠小。
但 XY 還是公尺座標，需要縮放、反轉軸向與平移，才成為圖片像素。
令 r 為公尺／像素，讓前方朝上、左方朝左：

```text
u = (y_max - y) / r
v = (x_max - x) / r

[u, v, 1]ᵀ = A @ [x, y, z, 1]ᵀ

A = [  0   -1/r  0  y_max/r ]
    [-1/r    0   0  x_max/r ]
    [  0     0   0     1   ]
```

A 的 Z 欄全部為零，最後一欄負責平移。最後一項固定為 1，沒有透視除法。

預設 x=(-10,70)、y=(-40,40)、r=0.1，圖像為 800×800：

| 點 (x,y,z) | 像素 (u,v) | 意義 |
|---|---|---|
| (0,0,0) | (400,700) | 感測器原點 |
| (20,0,0) | (400,500) | 前方 20 公尺 |
| (20,10,3) | (300,500) | 同時往左 10 公尺 |

把連續像素 floor 成整數格，再檢查 `0 <= u < width`、`0 <= v < height`。
這個定義包含 x_max/y_max 邊界、排除 x_min/y_min，避免最高索引超出圖片。

### Z 沒有完全丟掉

投影位置不使用 Z，但格子的值仍可使用高度、密度或反射強度。
目前先篩選 z_range 內的點，再對同一格取最高 Z：

```text
height[v, u] = max(所有落在此格且通過高度篩選的點的 z)
```

`np.maximum.at` 可處理重複索引，結果不依賴輸入點的順序。
沒有點的格子用 -inf 表示，不能填 0，因為 0 是真實的感測器高度。
最高點可能是車頂或樹枝，因此這是高度圖，不是地面估計，也不是語意分割。

預設保留 -3～3 公尺，低點偏藍、高點偏紅。白色原點標記與紅色 +X、
綠色 +Y 箭頭另外畫上，幫助理解位置。BEV 顏色代表高度，2D/3D 代表距離。
對應實作：`BEVRenderer.projection`、`rasterize`、`render`。

## Runtime 與 Qt signals/slots

啟動時勾選要開的 viewer，不需要事先產生 MP4。
每個播放步驟由背景 worker 讀一個點雲，計算所選視圖，再送回 GUI。
GUI 主類處理元件和顯示，功能類透過組合提供計算。

一次只允許一個工作；拖曳時丟棄過時結果，只顯示最新位置。
QImage 跨執行緒前 copy，NumPy 點資料送出後不再修改。
QPixmap、QWidget 與 OpenGL 元件只在 GUI thread 更新。
所有 viewer 共用播放位置與 FPS；關閉一個不會停止其他視窗。
FPS 是目標速度，計算太慢時會放慢而不跳幀。

資料格式依據本機 `devkit_tracking/devkit/readme.txt`，
以及 [KITTI 官方 Tracking 說明](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)。
