cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(NumpyCpp LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

add_executable(numpycpp_1)
target_sources(numpycpp_1
    PRIVATE
    numpycpp_1.cpp
)
target_link_libraries(numpycpp_1 PRIVATE Python3::Python Python3::Module Python3::NumPy)
