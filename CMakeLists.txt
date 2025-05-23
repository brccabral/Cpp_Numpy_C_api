cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(NumpyCpp LANGUAGES CXX)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
    set(Python3_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
endif ()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/venv")
    set(Python3_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv")
endif ()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

add_executable(numpycpp_1)
target_sources(numpycpp_1
    PRIVATE
    numpycpp_1.cpp
)
target_link_libraries(numpycpp_1 PRIVATE Python3::Python Python3::Module Python3::NumPy)

add_executable(numpycpp_2)
target_sources(numpycpp_2
    PRIVATE
    numpycpp_2.cpp
)
target_link_libraries(numpycpp_2 PRIVATE Python3::Python Python3::Module Python3::NumPy)

add_executable(numpycpp_3)
target_sources(numpycpp_3
    PRIVATE
    numpycpp_3.cpp
)
target_link_libraries(numpycpp_3 PRIVATE Python3::Python Python3::Module Python3::NumPy)
