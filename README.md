SparTile is a segmentation-free approach for analyzing spatila proteomics data. It uses sparse non-negative matrix factorization (sNMF) to encode protein marker and marker pair interations in multiplex protein images. Therefore, it is more interpretable than deep learning-based approaches. The helper functions are provided in the SparTile.py file. We use sklearn for sNMF implementation. The easiest way to use SparTile is to copy the main code (SparTile.py) to the active directoty. SparTile additionally uses:
numpy,
pandas,
scikit-image.

Please refer to the SparTileExample.ipynb for a minimal working example.

Reference: https://www.nature.com/articles/s43856-026-01400-4

Citation: Foroughi pour, A., Wu, TC., Noorbakhsh, J. et al. Prediction of outcome from spatial Protein profiling of triple-negative breast cancers. Commun Med (2026). https://doi.org/10.1038/s43856-026-01400-4

The raw IMC data of TNBCs used in the manuscript can be downloaded from: https://zenodo.org/records/17886494

Images of the IMC dataset for visualization of data can be downloaded from: https://zenodo.org/records/17056591
