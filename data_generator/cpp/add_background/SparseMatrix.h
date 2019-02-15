#pragma once
#include<iostream>  
#include<vector>  
#include<assert.h>  
using namespace std;

template<class T>
struct Triple
{
	int _row; //struct 默认为public
	int _col;
	double _dis;
	T _data;

	Triple(int row, int col, const T& data)
		:_row(row)
		, _col(col)
		, _data(data)
	{}
};

template<class T>
class SparseMatrix
{
protected:
	
	int _rows;
	int _cols;
	T _invalid;
public:
	vector<Triple<T> > _sm;//用 vector 容器 存放三元组  
	SparseMatrix(int rows, int cols,const T& invalid) 
	{ 
		this->_rows = rows;
		this->_cols = cols;
		this->_invalid = invalid; 
	};
	//************************************
	// Method:    根据矩阵构造一个稀疏矩阵
	// FullName:  SparseMatrix::SparseMatrix
	// Access:    public 
	// Returns:   
	// Qualifier:
	// Parameter: int * arr 输入矩阵
	// Parameter: int rows 矩阵的行
	// Parameter: int cols 矩阵的列
	// Parameter: const T & invalid 矩阵中的背景值
	//************************************
	SparseMatrix(int* arr, int rows, int cols, const T& invalid)
	{
		this->_rows = rows;
		this->_cols = cols;
		this->_invalid = invalid;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; ++j)
			{
				if (arr[i*cols + j] != invalid)//将有效值存储在一个一维数组中  
					this->_sm.push_back(Triple<T>(i, j, arr[i*cols + j]));//将三元组的无名对象push进去  
			}
		}
	};
	//************************************
	// Method:    向稀疏矩阵中添加值
	// FullName:  SparseMatrix<T>::add
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: int row 值所在的行
	// Parameter: int col 值所在的列
	// Parameter: const T& data 值
	//************************************
	void add(int row, int col, const T& data)
	{
		this->_sm.push_back(Triple<T>(row, col, data));
	};
 
	//************************************
	// Method:    访问稀疏矩阵中row行col中的元素
	// FullName:  SparseMatrix<T>::get
	// Access:    public 
	// Returns:   T& 得到的值
	// Qualifier:
	// Parameter: int row 值所在的行
	// Parameter: int col 值所在的列
	//************************************
	T& get(int row, int col)
	{
		size_t len = this->_sm.size();
		for (size_t i = 0; i < len; i++)
		{
			Triple<T> a = this->_sm[i];
			if (a._row == row &&a._col == col)//行列相等输出值  
				return a._data;
		}
		return _invalid;
	};

	//************************************
	// Method:    重载<<
	// FullName:  SparseMatrix<T>::operator<<
	// Access:    public 
	// Returns:   std::ostream&
	// Qualifier: //重载<<
	// Parameter: ostream & _cout
	// Parameter: SparseMatrix<T> & s
	//************************************
	friend ostream& operator<<(ostream& _cout, SparseMatrix<T>& s)
	{
		size_t idex = 0;
		for (size_t i = 0; i < s._rows; i++)
		{
			for (size_t j = 0; j < s._cols; j++)
			{
				if (idex < s._sm.size() && s._sm[idex]._row == i && s._sm[idex]._col == j)
				{
					_cout << s._sm[idex]._data << " ";
					++idex;
				}
				else
					_cout << s._invalid << " ";

			}
			_cout << endl;
		}
		return _cout;
	};

	//************************************
	// Method:    判断当前稀疏矩阵中是否包含某个值
	// FullName:  SparseMatrix<T>::contains
	// Access:    public 
	// Returns:   bool ture:包含 false:不包含
	// Qualifier:
	// Parameter: int row 值所在的行
	// Parameter: int col 值所在的列
	// Parameter: const T & data 值
	//************************************
	bool contains(int row, int col, const T& data)
	{
		size_t len = this->_sm.size();
		for (size_t i = 0; i < len; i++)
		{
			Triple<T> a = this->_sm[i];
			if (a._row == row && a._col == col && a._data == data)//行列相等输出值  
				return true;
		}
		return false;
	}
};