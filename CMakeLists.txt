cmake_minimum_required(VERSION 3.14)
project(mcts LANGUAGES CXX)

# Enable march=native and O2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Allow user override of build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Try to locate pybind11 installed in Python site-packages
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env python3 -c "import pybind11, sys; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 REQUIRED)
# Requires abseil built with CMake config mode (AbseilConfig.cmake)
find_package(absl REQUIRED CONFIG)
find_package(xsimd REQUIRED)

# Collect sources: pybind binding + all cpp sources in src/cpp (adjust if needed)
set(BINDING_SRC ${PROJECT_SOURCE_DIR}/src/bindings.cpp)
file(GLOB PROJECT_SRCS
     ${PROJECT_SOURCE_DIR}/src/*.cpp
     ${PROJECT_SOURCE_DIR}/src/*.cc
)

# Ensure binding source is present
list(FILTER PROJECT_SRCS INCLUDE REGEX ".*")
if(NOT EXISTS ${BINDING_SRC})
  message(FATAL_ERROR "Binding source not found: ${BINDING_SRC}")
endif()

# Remove binding from generic list to avoid double-adding
list(REMOVE_ITEM PROJECT_SRCS ${BINDING_SRC})

pybind11_add_module(mcts
    ${BINDING_SRC}
    ${PROJECT_SRCS}
)

target_include_directories(mcts PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
)

set_target_properties(mcts PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/zero"
)

target_link_libraries(mcts PRIVATE
    absl::flat_hash_map
    absl::flat_hash_set
    xsimd
)