project(llama LANGUAGES CXX CUDA)
#project(llama)

file(GLOB_RECURSE HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HOST_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE KERNEL_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)

add_executable(${PROJECT_NAME}
  ${HOST_FILES}
  ${HOST_HEADER_FILES}
  ${KERNEL_FILES}
  ${KERNEL_HEADER_FILES}
  ${PYTHON_FILES}
)
source_group("python scripts" FILES ${PYTHON_FILES})
source_group("cuda kernel" FILES ${KERNEL_FILES})
source_group("cuda header" FILES ${KERNEL_HEADER_FILES}) 

foreach(kernel ${KERNEL_HEADER_FILES})
  set_source_files_properties(${kernel} PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:$ENV{CUDA_PATH}/include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:$ENV{CUDA_PATH}/lib/x64/cudart.lib>
)