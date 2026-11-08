cmake_minimum_required(VERSION 3.12)
project(lam_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find Eigen3 (header-only)
find_package(Eigen3 REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

pybind11_add_module(lam_cpp
    admm_solver.cpp
    bindings.cpp
)

target_link_libraries(lam_cpp PRIVATE Eigen3::Eigen)
target_include_directories(lam_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
