cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(CDT)

SET(SRC 
    src/main.cpp 
    src/matrix.cpp 
    src/shape.cpp 
    src/dataset.cpp 
    src/layer.cpp 
    src/cdt.cpp 
    src/activation_function.cpp
    src/utils.cpp)

include_directories(include)
link_directories(lib)
add_executable(main ${SRC})
