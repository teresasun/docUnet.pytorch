DocUnetC.dll

DocUnetC.dll是一个用于将扫描图片进行折叠或扭曲后和背景相结合以生成拍照图像的c++动态链接库，不包含去黑色背景功能，黑色背景需要在原图上直接去掉

```c++
int generate_one_img_py(
    const char *img_path, 
    const char *background_path, 
    const char *save_path, 
    const int new_row,
    const int new_col,
    float vec_rows[],
    float vec_cols[], 
    int p_row, 
    int p_col, 
    float v_len, 
    float v_x_angle, 
    float v_z_angle, 
    int board, 
    int kernel, 
    float alpha
)
```



### Input requirements: 

```c++
v_len > 0
board >= 0
filter_pixel >= 0
alpha > 0
```



### Args:

- `img_path`: 扫描图像路径
- `background_path`: 背景图像路径
- `save_path`: 图像保存路径
- `new_row`: 图像被resize的新行大小
- `new_col`: 图像被resize的新列大小
- `vec_rows[]`: 存储行映射的数组
- `vec_cols[]`: 存储列映射的数组
- `p_row`:随机变形点的行坐标
- `p_col`:随机变形点的列坐标
- `v_len`: 变形的强度
- `v_x_angle`: 变形的x轴角度
- `v_z_angle`: 变形的z轴角度
- `board`: 背景图像加在原始图像上的边界
- `kernel`: 消除背景时邻域的大小
- `alpha`: 变形的传播程度,这个值越大，扭曲和折叠效果越全局，越小，扭曲和折叠效果越局部

### Returns:

一个代表是否成功的值，0：失败，1成功

# clip_img.dll
clip_img.dll是一个用于`去除扫描图像上黑色背景`和`统计边界处的黑色像素值占所有像素的占比`的动态链接库，其包含两个方法

1. 去除扫描图像上黑色背景

```c++
void clip_img(
	const char* img_path, 
	const char* background_path
)
```



### Args:

- `img_path`: 扫描图像路径
- `save_path`: 图像保存路径

### Returns:

`None`



2.  统计边界处的黑色像素值占所有像素的占比

```c++
float remove_img(
    const char* img_path，
    int val
)
```



### Args:

- `img_path`: 扫描图像路径
- `val`: 阈值，三通道均低于这个值的被判断为黑色像素

### Returns:

`float` 边缘处黑色像素占所有像素的比例