get_tilelangir_link_libraries(LIBS)
add_tilelangir_install_tool(tilelangir-compile
  tilelangir-compile.cpp
  PassPipeline.cpp
  DEPENDS ${LIBS} tilelangir_transforms
  LINK_LIBS tilelangir_transforms ${LIBS}
)
target_link_libraries(tilelangir-compile PRIVATE tilelangir_headers)

if(TILELANGIR_MLIR_LIBS)
  target_link_libraries(tilelangir-compile PRIVATE
    -Wl,--start-group
    ${TILELANGIR_MLIR_LIBS}
    -Wl,--end-group
    ${TILELANGIR_SYSTEM_LIBS}
  )
endif()

