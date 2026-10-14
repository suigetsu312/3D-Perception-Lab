# 三種 Viewer 的幾何理論

本專案使用 KITTI Tracking：每個序列包含逐幀相機影像、Velodyne
點雲、calibration 與 training 標註。Tracking 的 track ID 用來識別
跨影格的同一個物件，目前三種 viewer 都只顯示量測，尚未畫 tracking 方框。

## 共用的座標與資料

LiDAR 檔案每筆為四個 little-endian float32：`(x, y, z, reflectance)`。
第四項是反射強度，不是齊次座標！幾何運算先取 xyz，再補成 $`(x,y,z,1)^{\mathsf T}`$。

LiDAR 軸向為 +X 前方、+Y 左方、+Z 上方，單位公尺。
相機軸向為 +X 右方、+Y 下方、+Z 前方。兩者是不同的座標基底。
白色原點標記是 LiDAR 的 $`(0,0,0)`$，不是路面，也不是固定的全球位置。
目前沒有 OXTS 位姿補償，因此每幀仍使用當幀的感測器座標系。

公式用列向量 $`\mathbf p' = \mathbf T\mathbf p`$；NumPy 以每列儲存一個點，因此批次寫成
`points @ T.T`。矩陣乘法的順序由右往左讀。

## 齊次座標與剛體轉換

旋轉與平移原本是 $`\mathbf p' = \mathbf R\mathbf p + \mathbf t`$。補上齊次座標後可合成一次乘法：

```math
\mathbf T =
\begin{bmatrix}
\mathbf R_{3\times3} & \mathbf t_{3\times1} \\
\mathbf 0_{1\times3} & 1
\end{bmatrix}
```

```math
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix}
= \mathbf T
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
```

平移不影響方向向量，所以方向向量的第四項是 0，位置點則是 1。
這兩種不能混用。對應實作為 `geometry.to_homogeneous` 與 `transform_points`。

## 2D：LiDAR 疊到實體相機影像

序列 0000 的參數在 `data_tracking_calib/training/calib/0000.txt`。
整個序列共用 calibration，換序列時要重新讀取。

| 欄位 | 尺寸 | 功能 |
|---|---|---|
| `Tr_velo_cam` | $`3\times4`$ | 外參：LiDAR → 參考相機，前三欄 $`\mathbf R`$、最後一欄 $`\mathbf t`$ |
| `R_rect` | $`3\times3`$ | 參考相機 → 校正後相機座標 |
| `P2` | $`3\times4`$ | 校正後參考座標 → 左彩色相機 image_02 |

先將 `R_rect` 與 `Tr_velo_cam` 擴成 $`4\times4`$，再計算：

```math
\mathbf q = \mathbf P_2\mathbf R_{\mathrm{rect}}\mathbf T_{\mathrm{velo}\to\mathrm{cam}}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix},
\qquad
u = \frac{q_x}{q_z},\quad v = \frac{q_y}{q_z}
```

這裡 $`\mathbf q`$ 不是已經可以畫的像素，必須做透視除法。
簡化成針孔模型時：

```math
u = f_x\frac{X}{Z} + c_x,\qquad
v = f_y\frac{Y}{Z} + c_y
```

除以深度 $`Z`$ 造成近大遠小。相機後方、無效數值與影像外的點要排除。

$`\mathbf P_2`$ 的左側 $`3\times3`$ 是校正後內參 $`\mathbf K`$。序列 0000 的數值為：

```math
\mathbf K =
\begin{bmatrix}
721.5377 & 0 & 609.5593 \\
0 & 721.5377 & 172.8540 \\
0 & 0 & 1
\end{bmatrix}
```

$`f_x`$、$`f_y`$ 是像素單位的焦距，$`c_x`$、$`c_y`$ 是主點。但 $`\mathbf P_2`$ 的第四欄並不為零，
還包含彩色相機相對校正參考座標的平移效果，所以投影不能只用 $`\mathbf K`$。
提供的影像已 rectified，目前不再施加鏡頭畸變。

2D 顏色表示到 LiDAR 原點的歐氏距離 $`\sqrt{x^2+y^2+z^2}`$，不是相機 $`Z`$。
遠點先畫、近點後畫只是一個畫點策略，不是完整的相機遮擋處理。
對應實作：`KittiCamera`、`geometry.project_points`、`KittiRenderer`。

## 3D：自由移動的虛擬相機

3D 螢幕最後仍然是 2D 像素，使用的是虛擬相機：

```math
\mathbf p_{\mathrm{clip}} = \mathbf P\mathbf V\mathbf M\mathbf p_{\mathrm{local}},
\qquad
\mathbf p_{\mathrm{ndc}} =
\frac{1}{w_{\mathrm{clip}}}
\begin{bmatrix} x_{\mathrm{clip}} \\ y_{\mathrm{clip}} \\ z_{\mathrm{clip}} \end{bmatrix}
```

- Model（$`\mathbf M`$）：物件 → 世界。目前直接用 LiDAR xyz，所以 $`\mathbf M=\mathbf I_4`$。
- View（$`\mathbf V`$）：世界 → 虛擬相機。旋轉與平移改變觀看方向及中心。
- Projection（$`\mathbf P`$）：虛擬相機 → 裁切座標，包含 FOV、長寬比與裁切面。
- GPU 除以 $`w_{\mathrm{clip}}`$ 得到 NDC，再映射到螢幕像素。

這裡 $`\mathbf P`$ 不是 KITTI 的 $`\mathbf P_2`$，因此能從任意方向看整片點雲。
GPU 的深度測試讓同一像素的近點遮住遠點，但離散點之間仍然有空隙。
導航用的原點與 XYZ 軸刻意關閉深度測試，以免被點雲遮住。

逐幀只更新點的位置與顏色，不重設相機。因此可以邊旋轉邊播放。
3D 點大小以像素計算，與公尺大小的幾何物件不同。
操作及架構見 [3D viewer](3d-viewer.md)。

## BEV：XY 正交投影與柵格化

幾何上，BEV 就是忽略 Z、投到 XY 平面：

```math
\mathbf B = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix},
\qquad
\begin{bmatrix} x_{\mathrm{bev}} \\ y_{\mathrm{bev}} \end{bmatrix}
= \mathbf B\begin{bmatrix} x \\ y \\ z \end{bmatrix}
```

它是正交投影，不除以深度、不會近大遠小。
但 XY 還是公尺座標，需要縮放、反轉軸向與平移，才成為圖片像素。
令 $`r`$ 為公尺／像素，讓前方朝上、左方朝左：

```math
u = \frac{y_{\max}-y}{r},\qquad
v = \frac{x_{\max}-x}{r}
```

```math
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf A\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix},
\qquad
\mathbf A =
\begin{bmatrix}
0 & -\frac{1}{r} & 0 & \frac{y_{\max}}{r} \\
-\frac{1}{r} & 0 & 0 & \frac{x_{\max}}{r} \\
0 & 0 & 0 & 1
\end{bmatrix}
```

$`\mathbf A`$ 的 $`Z`$ 欄全部為零，最後一欄負責平移。最後一項固定為 1，沒有透視除法。

預設 $`x\in(-10,70]`$、$`y\in(-40,40]`$、$`r=0.1`$，圖像為 $`800\times800`$：

| 點 $`(x,y,z)`$ | 像素 $`(u,v)`$ | 意義 |
|---|---|---|
| $`(0,0,0)`$ | $`(400,700)`$ | 感測器原點 |
| $`(20,0,0)`$ | $`(400,500)`$ | 前方 20 公尺 |
| $`(20,10,3)`$ | $`(300,500)`$ | 同時往左 10 公尺 |

把連續像素取整數格，再檢查索引是否落在影像內：

```math
i=\lfloor u\rfloor,\qquad j=\lfloor v\rfloor
```

```math
0\le i<W,\qquad 0\le j<H
```

其中 $`W`$、$`H`$ 是影像寬高。對應的公尺座標範圍為：

```math
x_{\min}<x\le x_{\max},\qquad
y_{\min}<y\le y_{\max}
```

也就是包含上界、排除下界，避免最高索引超出圖片。

### Z 沒有完全丟掉

投影位置不使用 Z，但格子的值仍可使用高度、密度或反射強度。
目前先篩選 `z_range` 內的點，再對同一格取最高 $`Z`$。
令 $`\mathcal S_{j,i}`$ 表示通過高度篩選、且落在第 $`j`$ 列、第 $`i`$ 欄的點索引集合：

```math
\mathcal S_{j,i}=\left\lbrace k\mid
\lfloor u_k\rfloor=i,\;
\lfloor v_k\rfloor=j,\;
z_{\min}\le z_k\le z_{\max}
\right\rbrace
```

```math
h_{j,i} =
\begin{cases}
\displaystyle\max_{k\in\mathcal S_{j,i}} z_k, & \mathcal S_{j,i}\ne\varnothing, \\
-\infty, & \mathcal S_{j,i}=\varnothing.
\end{cases}
```

`np.maximum.at` 可處理重複索引，結果不依賴輸入點的順序。
沒有點的格子用 $`-\infty`$ 表示，不能填 0，因為 0 是真實的感測器高度。
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
