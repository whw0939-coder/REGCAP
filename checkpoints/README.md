# Checkpoints

No trained weights are distributed. `regcap.cli train` writes a best-validation-F1 checkpoint and a last checkpoint to its output directory. Use a checkpoint trained with this repository's 31-feature schema for `evaluate` and `predict`. Load checkpoint files only from trusted sources.
