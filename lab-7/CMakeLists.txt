cmake_minimum_required(VERSION 3.0.0)
project(cpp VERSION 0.1.0)

include(CTest)
enable_testing()

# set(Boost_INCLUDE_DIR /usr/include/boost)

find_package(OpenCV 4.5.2 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMVS REQUIRED)
find_package(Boost REQUIRED regex date_time system filesystem thread graph)

# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${BOOST_INCLUDE_DIRS}
add_executable(cpp main.cpp)
message(${OpenCV_INCLUDE_DIRS})
target_include_directories(cpp PUBLIC ${OpenCV_INCLUDE_DIRS} ${ceres_INCLUDE_DIRS} ${OpenMVS_INCLUDE_DIRS} ${eigen3_INCLUDE_DIRS} ${opengv_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(cpp ${OpenCV_LIBS})
target_link_libraries(cpp ${OpenMVS_LIBS})
target_link_libraries(cpp ${Boost_LIBRARIES})

set(GCC_COVERAGE_COMPILE_FLAGS "-lboost_filesystem -lboost_system")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include(CPack)

