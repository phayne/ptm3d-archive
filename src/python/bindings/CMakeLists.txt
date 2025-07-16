find_package(pybind11 REQUIRED)

pybind11_add_module(europa_illumination illumination_bindings.cpp)
target_link_libraries(europa_illumination PRIVATE illumination_model constants)

# Add the symlink command HERE after the target is defined
add_custom_command(
  TARGET europa_illumination POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    $<TARGET_FILE:europa_illumination>
    ${CMAKE_BINARY_DIR}/europa_illumination.so
  COMMENT "Creating symlink for Python module"
)