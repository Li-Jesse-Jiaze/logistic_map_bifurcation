# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(BifurcationCalc VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Add compiler optimization flags based on the compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # For GCC and Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
elseif(MSVC)
    # For Microsoft Visual C++
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Enable Link Time Optimization (LTO) if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)
if(LTO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find GMP and MPFR libraries
find_library(GMP_LIB gmp REQUIRED)      # Find GMP library
find_library(MPFR_LIB mpfr REQUIRED)    # Find MPFR library
# Find Eigen3 library
find_package(Eigen3 3.3 REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${GMP_LIB} ${MPFR_LIB})

# Add include directories for target
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# (Optional) Specify position-independent code if needed
# set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
