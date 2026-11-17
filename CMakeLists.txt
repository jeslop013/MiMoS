cmake_minimum_required(VERSION 3.20)
project(mimos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- C++ core (static library shared by the Python module and the CLI) -----------------
add_library(mimos_core STATIC
    cpp/src/network.cpp
    cpp/src/population.cpp
    cpp/src/model.cpp
    cpp/src/routing.cpp
    cpp/src/choice.cpp
    cpp/src/public_sim.cpp
    cpp/src/drive_sim.cpp
    cpp/src/memory.cpp
    cpp/src/io.cpp
)
target_include_directories(mimos_core PUBLIC cpp/include)
set_target_properties(mimos_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # /fp:precise is MSVC's default; stated explicitly because results must be reproducible
    # (no fused multiply-add contraction, no reassociation).
    target_compile_options(mimos_core PUBLIC /fp:precise /W3 /wd4996)
else()
    target_compile_options(mimos_core PUBLIC -ffp-contract=off -Wall)
endif()

# OpenMP parallelises route planning over agents. Optional: Apple clang ships without
# it (`brew install libomp` enables it); without OpenMP the code runs single-threaded with
# identical results.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mimos_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "MiMoS: OpenMP enabled")
else()
    message(STATUS "MiMoS: OpenMP not found, planning runs single-threaded")
endif()

# ---- Stand-alone CLI (profiling) --------------------------------------------------------
option(MIMOS_BUILD_CLI "Build the mimos_cli executable" ON)
if(MIMOS_BUILD_CLI)
    add_executable(mimos_cli cpp/main.cpp)
    target_link_libraries(mimos_cli PRIVATE mimos_core)
endif()

# ---- Python extension (only when built through scikit-build-core / pybind11) ------------
if(SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core cpp/bindings.cpp)
    target_link_libraries(_core PRIVATE mimos_core)
    install(TARGETS _core DESTINATION mimos)
endif()
