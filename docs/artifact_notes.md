# Artifact notes

The release provides the reference implementation of REGCAP described in the submitted TOSEM manuscript, including canonical 31-dimensional RDP and the main downstream pipeline. The current staging tree contains code and documentation only. It does not include datasets, Joern exports, Word2Vec models, processed pickles, or trained checkpoints.

The public workflow begins with prepared C functions and Joern JSON. Converting raw benchmark records into those files and training the dataset-specific Word2Vec model are external preparation steps. Record their versions and settings when comparing numbers with the paper.

The `smoke` command validates one sample with randomly initialized model weights. Classification scores from that command have no research meaning. Evaluation and region-level interpretation require a checkpoint trained with this 31-feature schema.

The license is pending author approval. Until the placeholder is replaced, repository publication and reuse permissions are unresolved.

Numerical paper results remain paper-reported unless an individual run is explicitly marked as rerun. Checkpoints must match the released architecture and 31-dimensional feature schema.
