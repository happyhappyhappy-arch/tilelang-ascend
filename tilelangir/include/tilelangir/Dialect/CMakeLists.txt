tilelangir_tablegen(tilelangir_dialect_decls_incgen
  TD_FILES TileLangIRDialect.td
  ARGS -gen-dialect-decls -dialect=tilelangir
)

tilelangir_tablegen(tilelangir_dialect_defs_incgen
  TD_FILES TileLangIRDialectDefs.td
  ARGS -gen-dialect-defs -dialect=tilelangir
)

tilelangir_tablegen(tilelangir_op_decls_incgen
  TD_FILES TileLangIROps.td
  ARGS -gen-op-decls
)

tilelangir_tablegen(tilelangir_op_defs_incgen
  TD_FILES TileLangIROpsDefs.td
  ARGS -gen-op-defs
)

tilelangir_tablegen(tilelangir_enum_decls_incgen
  TD_FILES TileLangIREnums.td
  ARGS -gen-enum-decls
)

tilelangir_tablegen(tilelangir_enum_defs_incgen
  TD_FILES TileLangIREnumsDefs.td
  ARGS -gen-enum-defs
)
