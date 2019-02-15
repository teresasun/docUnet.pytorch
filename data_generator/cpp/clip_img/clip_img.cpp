#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
extern "C" 
{
	void clip_img(const char* img_path, const char* save_path)
	{
		cv::Mat grey,show;
		cv::Mat img = cv::imread(img_path,1);
		if (img.channels()==3)
		{
			cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
		}
		cv::threshold(grey, grey, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::morphologyEx(grey, grey, cv::MORPH_OPEN, kernel);

		int rows = img.rows;
		int cols = img.cols;
		int temp_pixel,top,bottom,left,right;
		top = bottom = left = right = -1;
		//从左到右
		for (int col_j = 0; col_j < cols; col_j++)
		{
			for (int row_i = 0; row_i < rows; row_i++)
			{
				temp_pixel = (int)grey.at<uchar>(row_i, col_j);
				if (temp_pixel != 0)
				{
					left = col_j;
					break;
				}
			}
			if (left!=-1)
			{
				break;
			}
		}
		//从右到左
		for (int col_j = cols-1; col_j >= 0; col_j--)
		{
			for (int row_i = 0; row_i < rows; row_i++)
			{
				temp_pixel = (int)grey.at<uchar>(row_i, col_j);
				if (temp_pixel != 0)
				{
					right = col_j;
					break;
				}
			}
			if (right != -1)
			{
				break;
			}
		}
		//从上到下
		for (int row_i = 0; row_i < rows; row_i++)
		{
			for (int col_j = 0; col_j < cols; col_j++)
			{	
				temp_pixel = (int)grey.at<uchar>(row_i, col_j);
				if (temp_pixel != 0)
				{
					top = row_i;
					break;
				}	
			}
			if (top != -1)
			{
				break;
			}
		}
		//从下到上
		for (int row_i = rows -1; row_i >= 0; row_i--)
		{
			for (int col_j = 0; col_j < cols; col_j++)
			{
				temp_pixel = (int)grey.at<uchar>(row_i, col_j);
				if (temp_pixel != 0)
				{
					bottom = row_i;
					break;
				}
			}
			if (bottom != -1)
			{
				break;
			}
		}
		cv::Mat result = img(cv::Rect(left, top, right - left + 1, bottom - top + 1));
		cv::imwrite(save_path, result);
	}
}
extern "C"
{
    float remove_img(const char* img_path,int val)
	{
		cv::Mat img = cv::imread(img_path);
		int rows = img.rows;
		int cols = img.cols;
		int  top, bottom, left, right;
		top = bottom = left = right = -1;
		cv::Vec3b temp_pixel;
		float count = 0.0;
		vector<vector<int> > p(rows, vector<int>(cols, 0));
		for (int col_j = 0; col_j < cols; col_j++)
		{
			int row_i;
			//从上到下
			for (row_i = 0; row_i < rows; row_i++)
			{
				temp_pixel = img.at<cv::Vec3b>(row_i, col_j);
				if (temp_pixel[0] <= val && temp_pixel[1] <= val && temp_pixel[2] <= val && p[row_i][col_j] != 1)
				{
					p[row_i][col_j] = 1;
					count++;
				}
				else
				{
					break;
				}
			}
			//从下到上
			for (int temp_row_i = rows - 1; temp_row_i > row_i; temp_row_i--)
			{
				temp_pixel = img.at<cv::Vec3b>(temp_row_i, col_j);
				if (temp_pixel[0] <= val && temp_pixel[1] <= val && temp_pixel[2] <= val && p[temp_row_i][col_j] != 1)
				{
					p[temp_row_i][col_j] = 1;
					count++;
				}
				else
				{
					break;
				}
			}
		}
		for (int row_i = 0; row_i < rows; row_i++)
		{
			int col_j;
			//从上到下
			for (col_j = 0; col_j < cols; col_j++)
			{
				temp_pixel = img.at<cv::Vec3b>(row_i, col_j);
				if (temp_pixel[0] <= val && temp_pixel[1] <= val && temp_pixel[2] <= val && p[row_i][col_j] != 1)
				{
					p[row_i][col_j] = 1;
					count++;
				}
				else
				{
					break;
				}
			}
			//从下到上
			for (int temp_col_j = cols - 1; temp_col_j > col_j; temp_col_j--)
			{
				temp_pixel = img.at<cv::Vec3b>(row_i, temp_col_j);
				if (temp_pixel[0] <= val && temp_pixel[1] <= val && temp_pixel[2] <= val && p[row_i][temp_col_j] != 1)
				{
					p[row_i][temp_col_j] = 1;
					count++;
				}
				else
				{
					break;
				}
			}
		}
		return count / (rows*cols);
	}
}

extern "C"
{
    float remove_img_thre(const char* img_path)
    {

        cv::Mat img = cv::imread(img_path, 0);
        cv::threshold(img, img, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(img, img, cv::MORPH_OPEN, element);
        int rows = img.rows;
        int cols = img.cols;
        int  top, bottom, left, right;
        top = bottom = left = right = -1;
        int temp_pixel;
        float count = 0.0;
        vector<vector<int> > p(rows, vector<int>(cols, 0));
        for (int col_j = 0; col_j < cols; col_j++)
        {
            int row_i;
            //从上到下
            for (row_i = 0; row_i < rows; row_i++)
            {
                temp_pixel = (int)img.at<uchar>(row_i, col_j);
                if (temp_pixel == 0 && p[row_i][col_j] != 1)
                {
                    p[row_i][col_j] = 1;
                    count++;
                }
                else
                {
                    break;
                }
            }
            //从下到上
            for (int temp_row_i = rows - 1; temp_row_i > row_i; temp_row_i--)
            {
                temp_pixel = (int)img.at<uchar>(temp_row_i, col_j);
                if (temp_pixel == 0 && p[temp_row_i][col_j] != 1)
                {
                    p[temp_row_i][col_j] = 1;
                    count++;
                }
                else
                {
                    break;
                }
            }
        }
        for (int row_i = 0; row_i < rows; row_i++)
        {
            int col_j;
            //从上到下
            for (col_j = 0; col_j < cols; col_j++)
            {
                temp_pixel = (int)img.at<uchar>(row_i, col_j);
                if (temp_pixel == 0 && p[row_i][col_j] != 1)
                {
                    p[row_i][col_j] = 1;
                    count++;
                }
                else
                {
                    break;
                }
            }
            //从下到上
            for (int temp_col_j = cols - 1; temp_col_j > col_j; temp_col_j--)
            {
                temp_pixel = (int)img.at<uchar>(row_i, temp_col_j);
                if (temp_pixel == 0 && p[row_i][temp_col_j] != 1)
                {
                    p[row_i][temp_col_j] = 1;
                    count++;
                }
                else
                {
                    break;
                }
            }
        }
        return count / (rows*cols);
    }
}

int main()
{
	clip_img("/data/zj/docUnet/cpp/clip_img/1_1.jpg", "/data/zj/docUnet/cpp/clip_img/1_result.jpg");
	//system("pause");
    return 0;
}

