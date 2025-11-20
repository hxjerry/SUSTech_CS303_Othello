cmake_minimum_required(VERSION 3.14)
project(ZeroTest CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -O2 -march=native)
endif()

# Find and link Abseil
find_package(absl REQUIRED)

# Find and link xsimd
find_package(xsimd REQUIRED)

# Define source files
# Assuming the structure based on your Makefile:
# main.cpp is in the current directory
# bitboard.cpp, zorbist.cpp, mcts.cpp, transposition_table.cpp are in the parent directory
set(SOURCES
    main.cpp
    ../bitboard.cpp
    ../mcts.cpp
    ../transposition_table.cpp
    ../zorbist.cpp
)

# Create executable
add_executable(main ${SOURCES})

# Include directories
target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/.. # To include headers from src/cpp
)

# Link libraries
target_link_libraries(main PRIVATE 
    absl::flat_hash_map  # Example abseil component, add others as needed
    absl::strings
    xsimd
)