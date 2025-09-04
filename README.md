SparTile is a segmentation-free approach for analyzing spatila proteomics data. It uses sparse non-negative matrix factorization (sNMF) to encode protein marker and marker pair interations in multiplex protein images. Therefore, it is more interpretable than deep learning-based approaches. The helper functions are provided in the SparTile.py file. We use sklearn for sNMF implementation. The easiest way to use SparTile is to copy the main code (SparTile.py) to the active directoty. SparTile additionally uses:
numpy,
pandas,
scikit-image.

Please refer to the SparTileExample.ipynb for a miniml working example.

Reference: https://www.biorxiv.org/content/10.1101/2025.04.18.649541v1

The TNBC dataset used in the manuscript can be downloaded from: to be added.

Images of the IMC dataset for visualization of data can be downloaded from: https://zenodo.org/records/17056591
