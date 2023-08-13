cmake_minimum_required(VERSION 3.27.0)

project(vplan_analysis)

message(STATUS "Using LLVM in: ${LLVM_DIR}")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using Cmake LLVM in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

#option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})

set(CMAKE_COMPILE_WARNING_AS_ERROR True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-fproc-stat-report" COMPILER_SUPPORTS_PROC_STAT_REPORT)
if(COMPILER_SUPPORTS_PROC_STAT_REPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fproc-stat-report=abc")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)

