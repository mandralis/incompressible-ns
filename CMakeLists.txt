cmake_minimum_required(VERSION 3.17)
project(incompressible_ns)

INCLUDE_DIRECTORIES(/opt/homebrew/opt/fftw/include)
INCLUDE_DIRECTORIES(/Users/imandralis/git/eigen-3.4.0)
INCLUDE_DIRECTORIES(./numerical-recipes)
INCLUDE_DIRECTORIES(./steppers)
INCLUDE_DIRECTORIES(./io-tools)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES main.cpp io-tools/eigen-io.h)

add_executable(incompressible_ns ${SOURCE_FILES})