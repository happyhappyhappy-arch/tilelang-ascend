add_library(tilelangir_transforms STATIC
  CVAnnotate.cpp
  SimpleMultiBuffer.cpp
  AnalyzeCrossScope.cpp
  MaterializeWorkspace.cpp
  InsertWorkspace.cpp
  InjectBlockSync.cpp
  InsertVID.cpp
  OutlineScope.cpp
  EmitHostCallbacks.cpp
)
set_target_properties(tilelangir_transforms PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(tilelangir_transforms PRIVATE -frtti)
add_dependencies(tilelangir_transforms tilelangir_transforms_incgen)
target_link_libraries(tilelangir_transforms PUBLIC tilelangir_headers tilelangir_dialect)
