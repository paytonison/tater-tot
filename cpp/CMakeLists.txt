cmake_minimum_required(VERSION 3.20)

project(tater_tot VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tater_tot
    src/checkpoint.cpp
    src/data.cpp
    src/model.cpp
    src/optimizer.cpp
    src/tensor.cpp
)

target_include_directories(tater_tot
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(tater_tot PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

add_executable(tater_train examples/train.cpp)
target_link_libraries(tater_train PRIVATE tater_tot)

add_executable(tater_generate examples/generate.cpp)
target_link_libraries(tater_generate PRIVATE tater_tot)

add_executable(tater_tests tests/test_main.cpp)
target_link_libraries(tater_tests PRIVATE tater_tot)

enable_testing()
add_test(NAME tater_tests COMMAND tater_tests)

