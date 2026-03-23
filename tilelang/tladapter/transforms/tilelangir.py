# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TileLangIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

_F = "func.func"

cv_split = pass_fn("tilelangir-cv-split")
vectorize = pass_fn("tilelangir-vectorize")

cv_annotate = pass_fn("tilelangir-cv-annotate")
insert_vid = pass_fn("tilelangir-insert-vid", anchor=_F)
analyze_cross_scope = pass_fn("tilelangir-analyze-cross-scope")
materialize_workspace = pass_fn("tilelangir-materialize-workspace")
simple_multibuffer = pass_fn("tilelangir-simple-multibuffer", anchor=_F)
inject_block_sync = pass_fn("tilelangir-inject-block-sync", anchor=_F)
outline_scope = pass_fn("tilelangir-outline-scope")
emit_host_callbacks = pass_fn("tilelangir-emit-host-callbacks")
