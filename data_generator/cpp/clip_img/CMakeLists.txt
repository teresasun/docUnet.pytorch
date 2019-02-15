cmake_minimum_required(VERSION 3.5)
project( clip_img )# 项目名称
find_package( OpenCV REQUIRED )# 找到opencv源
# add_executable( clip_img clip_img.cpp )# 指定生成目标
add_library(clip_img SHARED clip_img.cpp)
target_link_libraries( clip_img ${OpenCV_LIBS} )# 添加opencv链接选项