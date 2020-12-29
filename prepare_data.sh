#!bash

python make_model.py
python -m spacy init-model en ./data/models/crit2vec_vox-machina --vectors-loc ./data/crit2vec_vox-machina.txt
python -m spacy init-model en ./data/models/crit2vec_mighty-nein --vectors-loc ./data/crit2vec_mighty-nein.txt
python -m spacy init-model en ./data/models/crit2vec_full-model --vectors-loc ./data/crit2vec_full-model.txt
python prepare_umap.py
