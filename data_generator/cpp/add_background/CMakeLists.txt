cmake_minimum_required(VERSION 3.5)
project( add_background )# 项目名称
find_package( OpenCV REQUIRED )# 找到opencv源

# add_executable( add_background DocUnetC.cpp )# 指定生成目标
add_library(add_background SHARED DocUnetC.cpp)
target_link_libraries( add_background ${OpenCV_LIBS} )# 添加opencv链接选项
