add_library(tilelangir_transforms STATIC
  CVSplit.cpp
  Vectorize.cpp
)
target_compile_options(tilelangir_transforms PRIVATE -frtti)
add_dependencies(tilelangir_transforms tilelangir_transforms_incgen)
target_link_libraries(tilelangir_transforms PUBLIC tilelangir_headers)
