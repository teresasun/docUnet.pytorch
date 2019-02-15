/*
根据扫描文档生成扭曲和折叠的文档
需要人为设定的值
V._strength 表示像素点偏移的强度
alpha 这个值越大，扭曲和折叠效果越全局，越小，扭曲和折叠效果越局部
*/
#include <iostream>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#define random_int(x) (rand() % x)
#define random_float() (rand() / double(RAND_MAX))

using namespace std;
using namespace cv;

int row_top_move;    //点向上移动的边界，大于这个边界的点都没有移动
int col_left_move;   //点向左移动的边界，大于这个边界的点都没有移动
int row_bottom_move; //点向下移动的边界，小于这个边界的点都没有移动
int col_right_move;  //点向右移动的边界，小于这个边界的点都没有移动
int min_row;         //点移动之后在构造的大图上的最小行
int min_col;         //点移动之后在构造的大图上的最小列

struct V
{
    int _strength;
    float _x_angle; // 和x轴之间的夹角
    float _z_angle; // 和z轴之间的夹角
    float _x;       // x轴方向的长度
    float _y;       // y轴方向的长度
    float _z;       // z轴方向的长度

    V(float x_angle, float z_angle, const float strength)
        : _x_angle(x_angle), _z_angle(z_angle), _strength(strength)
    {
        this->init();
    };

    void init()
    {
        this->_z = this->_strength * sin(this->_z_angle);
        double x_y_strength = this->_strength * cos(this->_z_angle);
        this->_x = x_y_strength * cos(this->_x_angle);
        this->_y = x_y_strength * sin(this->_x_angle);
    }

    friend ostream &operator<<(ostream &_cout, V &v)
    {
        return _cout << "_x_angle:" << v._x_angle << ",_z_angle:" << v._z_angle << ",_strength:" << v._strength
                     << ",x:" << v._x << ",y:" << v._y << ",z:" << v._z;
    };
};

//************************************
// Method:    计算空间内点到直线的3D距离
// FullName:  dis_p_l
// Access:    public
// Returns:   double 空间内点到直线的距离
// Qualifier:
// Parameter: const V & v 空间直线的方向向量
// Parameter: const Point & p 空间直线经过的点
// Parameter: int row 另一点的行索引 y
// Parameter: int col 另一点你的列索引 y
//************************************
double dis_p_l(const V &v, const Point &p, int row, int col)
{
    double k = (v._x * (col - p.x) + v._y * (row - p.y)) / (pow(v._x, 2) + pow(v._y, 2) + pow(v._z, 2));
    double dis = sqrt(pow(v._x * k + p.x - col, 2) + pow(v._y * k + p.y - row, 2) + pow(v._z * k, 2));
    return dis;
}

void get_dis(vector<double> &dis_list,const int rows,const int cols,const V &v, const Point &p,double &max_dis)
{
    for (int row_i = 0; row_i < rows; row_i++)
    {
        for (int col_j = 0; col_j < cols; col_j++)
        {
            double dis = dis_p_l(v, p, row_i, col_j);
            dis_list.push_back(dis);
            max_dis = max(dis, max_dis);
        }
    }
}

void curves_folds_image(Mat new_img,const Mat img,float vec_rows[], float vec_cols[],vector<vector<int> >&new_row_list,vector<vector<int> >&new_col_list,const int n,const int curves_folds_num,const bool is_curves,int &all_num,const int index,int &max_row,int &max_col,const int new_img_row,const int new_img_col,const int relax)
{
    int rows = img.rows;
    int cols = img.cols;

    for (int curves_folds_i = 0; curves_folds_i < curves_folds_num; curves_folds_i++)
    {
        // 初始化随机种子
        srand((index + all_num) * (unsigned)time(NULL));
        //随机p和v
        int p_row = random_int(new_img_row);
        int p_col = random_int(new_img_col);
        float alpha = random_float() + 0.1;
        int v_len = (random_float() + 1) * alpha * 70;
        float v_x_angle = random_float() * 2 * 3.14159;
        int v_z_angle = 0;
        Point p(p_col, p_row);
        V v(v_x_angle, v_z_angle, v_len);

        // 计算距离
        double max_dis = 0;
        vector<double> dis_list;
        get_dis(dis_list,rows,cols,v,p,max_dis);

        int i = 0;
        for (int row_i = 0; row_i < rows; row_i++)
        {
            for (int col_j = 0; col_j < cols; col_j++)
            {
                // 添加折叠效果
                double dis = dis_list[i] / max_dis;
                double w;
                if (is_curves){
                    w = 1 - pow(dis, alpha); //扭曲
                }else{
                    w = alpha / (dis + alpha); //折叠
                }
                // 将x,y方向的变化量存储起来，在最后一次变换时才执行赋值
                new_row_list[row_i][col_j] += w * v._y;
                new_col_list[row_i][col_j] += w * v._x;
                if (all_num == n - 1)
                {
                    int new_row = new_row_list[row_i][col_j] + relax / 2 + row_i;
                    int new_col = new_col_list[row_i][col_j] + relax / 2 + col_j;
                    if (row_i == 0)
                    {
                        row_top_move = max(new_row, row_top_move);
                    }
                    else if (row_i == rows - 1)
                    {
                        row_bottom_move = min(new_row, row_bottom_move);
                    }
                    if (col_j == 0)
                    {
                        col_left_move = max(new_col, col_left_move);
                    }
                    else if (col_j == cols - 1)
                    {
                        col_right_move = min(new_col, col_right_move);
                    }
                    min_row = min(min_row, new_row);
                    max_row = max(max_row, new_row);
                    min_col = min(min_col, new_col);
                    max_col = max(max_col, new_col);
                    new_img.at<Vec3b>(new_row, new_col) = img.at<Vec3b>(row_i, col_j);
                    vec_rows[row_i * cols + col_j] = new_row;
                    vec_cols[row_i * cols + col_j] = new_col;
                }
                i++;
            }
        }
        all_num ++;
    }
}

//************************************
// Method:    为图像添加扭曲和折叠效果
// FullName:  folds_curves_img_vector
// Access:    public
// Returns:   Mat 添加扭曲和折叠之后的图片
// Qualifier:
// Parameter: const Mat img 原图像
// Parameter: const V & v 存储了直线方向和强度的结构体
// Parameter: const Point & p 选取的起始扭曲点
// Parameter: const int alpha 控制效果传播距离的变量，越大，效果越全局；越小，效果越局部
//************************************
Mat folds_curves_img_vector(const Mat img, float vec_rows[], float vec_cols[], const int new_img_row, const int new_img_col, const int n, int index)
{
    int rows = img.rows;
    int cols = img.cols;
    int relax = 200 * 4 * n;
    Mat new_img = Mat(rows + relax, cols + relax, img.type(), Scalar(0, 255, 0));

    min_row = rows + relax;
    min_col = cols + relax;
    int max_row = 0; // 变换之后的最大行
    int max_col = 0; // 变换之后的最大列

    col_left_move = 0;
    row_top_move = 0;
    col_right_move = cols + relax + 2;
    row_bottom_move = rows + relax + 2;

    vector<vector<int> > new_row_list(rows, vector<int>(cols, 0));
    vector<vector<int> > new_col_list(rows, vector<int>(cols, 0));

    int curves_num = round(n * 0.3); //计算扭曲次数
    int folds_num = round(n * 0.7);  //计算折叠次数
    // cout<<n<<" "<<curves_num<<" "<<folds_num<<endl;
    int all_num = 0;
    // 开始扭曲
    curves_folds_image(new_img,img,vec_rows,vec_cols,new_row_list,new_col_list,n,curves_num,true,all_num,index,max_row,max_col,new_img_row,new_img_col,relax);
    //开始折叠
    curves_folds_image(new_img,img,vec_rows,vec_cols,new_row_list,new_col_list,n,folds_num,false,all_num,index,max_row,max_col,new_img_row,new_img_col,relax);
    
    new_img = new_img(Rect(min_col, min_row, max_col - min_col + 1, max_row - min_row + 1));
    
    // row_top_move -= min_row;
    // col_left_move -= min_col;
    // row_bottom_move = max_row - row_bottom_move;
    // col_right_move = max_col - col_right_move;

    // cout<<min_col<<" "<<min_row<<" "<<max_col - min_col + 1<<" "<<max_row - min_row + 1<<endl;
    return new_img;
}

//************************************
// Method:    对像素进行插值
// FullName:  linear_interp
// Access:    public
// Returns:   Mat
// Qualifier:
// Parameter: Mat img 生成的图像
// Parameter: int field_size 插值的邻域 经过测试，5为最佳值
//************************************
Mat linear_interp(Mat img, int field_size)
{
    int start = -field_size / 2;
    int end = field_size / 2 + 1;
    Mat dstImage = img.clone();
    int rows = dstImage.rows;
    int cols = dstImage.cols;

    Vec3b temp_pixel;

    for (int row_i = abs(start); row_i < rows - end; row_i++)
    {
        for (int col_j = abs(start); col_j < cols - end; col_j++)
        {
            temp_pixel = dstImage.at<Vec3b>(row_i, col_j);
            if (temp_pixel[0] <= 50 && temp_pixel[1] >= 200 && temp_pixel[2] <= 50)
            {
                int num, a, b, c;
                num = a = b = c = 0;
                Vec3b temp_pixel;
                // 取8邻域内不为绿色像素的均值
                for (int i = start; i < end; i++)
                {
                    for (int j = start; j < end; j++)
                    {
                        temp_pixel = img.at<Vec3b>(row_i + i, col_j + j);
                        if (temp_pixel[0] > 50 || temp_pixel[1] < 200 || temp_pixel[2] > 50)
                        {
                            a += temp_pixel[0];
                            b += temp_pixel[1];
                            c += temp_pixel[2];
                            num++;
                        }
                    }
                }
                if (num > field_size * field_size / 2)
                {
                    dstImage.at<Vec3b>(row_i, col_j) = Vec3b(a / num, b / num, c / num);
                }
            }
        }
    }
    return dstImage;
}

void warp_images(Mat new_img,const Mat img,float vec_rows[], float vec_cols[],const int index)
{
    // 初始化随机种子
    srand(index * (unsigned)time(NULL));
    int rows = new_img.rows;
    int cols = new_img.cols;
    
    // 构造执行仿射的四个点
    Point2f p1 = Point2f(0, 0);  // col,row
    Point2f p2 = Point2f(cols - 1,0);  // col,row
    Point2f p3 = Point2f(0, rows - 1);
    Point2f p4 = Point2f(cols - 1, rows - 1);
    Point2f src_points[] = {p1, p2, p3, p4};

    int p1_dx = rand() % (30);
    int p1_dy = rand() % (50);
    p1.x += p1_dx;
    p1.y += p1_dy;

    p2.x -= p1_dx;
    p2.y += p1_dy;


    int p3_dx = rand() % (10);
    int p3_dy = rand() % (10);
    p3.x += p3_dx;
    p3.y -= p3_dy;

    p4.x -= p3_dx;
    p4.y -= p3_dy;
    Point2f dst_points[] = {p1, p2, p3, p4};
    Mat M = getPerspectiveTransform(src_points, dst_points);
    // 对图像进行透视变换
	warpPerspective(new_img, new_img, M, cv::Size(cols,rows), cv::INTER_LINEAR,BORDER_CONSTANT,cv::Scalar(0,255,0));

    // 将变换后的x,y坐标加到列表里
    rows = img.rows;
    cols = img.cols;
    vector<Point2f> points, points_trans;
    for (int row_i = 0; row_i < rows; row_i++)
    {
        for (int col_j = 0; col_j < cols; col_j++)
        {
            points.push_back(Point2f(vec_cols[row_i * cols + col_j] - min_col,vec_rows[row_i * cols + col_j] - min_row));
        }
    }
    // 对点进行相应的变换
    perspectiveTransform(points, points_trans, M);

    min_row = rows;
    min_col = cols;
    for (int row_i = 0; row_i < rows; row_i++)
    {
        for (int col_j = 0; col_j < cols; col_j++)
        {
            int new_row =  points_trans[row_i * cols + col_j].y;
            int new_col =  points_trans[row_i * cols + col_j].x;
            vec_cols[row_i * cols + col_j] = new_col;
            vec_rows[row_i * cols + col_j] = new_row;
            min_row = min(min_row, new_row);
            min_col = min(min_col, new_col);
            if (row_i == 0)
            {
                row_top_move = max(new_row, row_top_move);
            }
            else if (row_i == rows - 1)
            {
                row_bottom_move = min(new_row, row_bottom_move);
            }
            if (col_j == 0)
            {
                col_left_move = max(new_col, col_left_move);
            }
            else if (col_j == cols - 1)
            {
                col_right_move = min(new_col, col_right_move);
            }
        }
    }
}

Mat add_background(Mat img, const string &background_path, int board)
{
    int rows = img.rows;
    int cols = img.cols;
    //int board = 50;
    Mat background_image = imread(background_path);
    //图片在背景图中放置的位置-先设为中间位置
    resize(background_image, background_image, Size(cols + board * 2, rows + board * 2));
    Vec3b temp_pixel;
    // col_right_move = cols - col_right_move;
    // row_bottom_move = rows - row_bottom_move;
    int row_start = board;
    int col_start = board;
    // 从上到下
    for (int col_j = 0; col_j < cols; col_j++)
    {
        for (int row_i = 0; row_i < rows; row_i++)
        {
            temp_pixel = img.at<Vec3b>(row_i, col_j);
            if (col_j > col_left_move + 10 && col_j < col_right_move - 10 && row_i > row_top_move + 10 && row_i < row_bottom_move - 10)
            {
                background_image.at<Vec3b>(row_i + row_start, col_j + col_start) = temp_pixel;
            }
            else
            {
                if (temp_pixel[0] == 0 && temp_pixel[1] == 255 && temp_pixel[2] == 0)
                {
                    continue;
                }
                else
                {
                    background_image.at<Vec3b>(row_i + row_start, col_j + col_start) = temp_pixel;
                }
            }
        }
    }
    return background_image;
}

//************************************
// Method:    生成一张图片的py接口
// FullName:  generate_one_img_py
// Access:    public
// Returns:   DLLEXPORT int
// Qualifier:
// Parameter: const char * img_path 扫描图像路径
// Parameter: const char * background_path 背景图像路径
// Parameter: const char * save_path 图像保存路径
// Parameter: const int new_row 图片resize的行大小
// Parameter: const int new_col 图片resize的列大小
// Parameter: float vec_rows[] 行方向的映射矩阵
// Parameter: float vec_cols[] 列方向的映射矩阵
// Parameter: int n 执行扭曲和矫正的总次数
// Parameter: int index 控制随机数的种子
// Parameter: int kernel 用于插值的领域大小，背景像素的值去这个领域的均值
//************************************
extern "C"
{
    int generate_one_img_py(const char *img_path, const char *background_path, const char *save_path, const int new_row, const int new_col, float vec_rows[],
                            float vec_cols[], int n, int index, int kernel)
    {
        Mat src = imread(img_path);
        if (src.empty())
        {
            return 0;
        }
        resize(src, src, Size(new_col, new_row));
        // 开始生成图片
        Mat new_img = folds_curves_img_vector(src, vec_rows, vec_cols, new_row, new_col, n, index);
        // 开始插值
        Mat dst_img = linear_interp(new_img, kernel);
        //执行透视变换
        warp_images(dst_img,src,vec_rows,vec_cols,index);
        // 开始添加背景
        int board = 0;// rand() % (100);
        Mat final_img = add_background(dst_img, background_path, board);
        int rows = src.rows;
        int cols = src.cols;

        // 计算映射在缩放图像上的新映射
        double ratio_row = rows * 1.0 / final_img.rows;
        double ratio_col = cols * 1.0 / final_img.cols;
        for (int row_i = 0; row_i < rows; row_i++)
        {
            for (int col_j = 0; col_j < cols; col_j++)
            {
                vec_rows[row_i * cols + col_j] = (vec_rows[row_i * cols + col_j] + board) * ratio_row;
                vec_cols[row_i * cols + col_j] = (vec_cols[row_i * cols + col_j] + board) * ratio_col;
            }
        }
        //将图片resize到和原图一样大，以保证gt和网络输出可以进行比较
        resize(final_img, final_img, src.size());
        imwrite(save_path, final_img);

        return 1;
    }
}

int main()
{
    //    int a = generate_one_img_py("/home/zj/temp/0/0_24.jpg", "/home/zj/Desktop/bg/blotchy_0003.jpg","1.jpg",100,50,5,1);
    int **rows, **cols;
    rows = (int **)malloc(sizeof(int *) * 1547);
    cols = (int **)malloc(sizeof(int *) * 1547);
    for (int i = 0; i < 1547; i++)
    { // 按行分配每一列
        rows[i] = (int *)malloc(sizeof(int) * 2440);
        cols[i] = (int *)malloc(sizeof(int) * 2440);
    }
    // generate_one_img("/home/zj/temp/0/0_24.jpg", "/home/zj/Desktop/bg/blotchy_0003.jpg", rows, cols);

    // waitKey();
    return 0;
}