cmake_minimum_required(VERSION 3.22.1)

project("resize")

# trim.cpp uses std::clamp and std::thread.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(resize SHARED
    hdr.cpp
    resize.cpp
    rotate.cpp
    trim.cpp
    window.cpp
)

target_link_libraries(resize android jnigraphics log)
