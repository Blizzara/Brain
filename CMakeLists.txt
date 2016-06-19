cmake_minimum_required(VERSION 2.8)
project(brain)


add_executable(brain main.cpp layer.cpp neuralnetwork.cpp reader.cpp)

set(CMAKE_CXX_FLAGS "-std=c++0x")
#set(CMAKE_EXE_LINKER_FLAGS "-static -static-libstdc++ -static-libgcc")
set(CMAKE_VERBOSE_MAKEFILE 1 )
#set(CMAKE_CXX_COMPILER "g++-4.6")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

install(TARGETS brain RUNTIME DESTINATION bin)
