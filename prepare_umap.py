import json
from itertools import product
from pathlib import Path
from string import punctuation

import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm, trange
from umap import UMAP

from crit2vec import lookup_index


def get_relevant_vectors(words_by_frequency, keep_words, nlp, n_words):
    # Get the first n_words word vectors that satsify some constraints
    cast = {"matt", "liam", "laura", "travis", "talesin", "marisha", "ashley"}
    vox_machina = {"allura", "kima", "gilmore", "vax", "vex", "trinket", "grog", "percy", "keyleth", "pike"}
    mighty_nein = {"pumat", "nott", "caleb", "jester", "fjord", "molly", "caduceus", "beau", "yasha"}
    start_words = cast | vox_machina | mighty_nein
    
    keep_words = set(keep_words)
    keep_words -= start_words
    top_words = list(start_words)
    top_vectors = [nlp(word).vector for word in top_words]
    for word in words_by_frequency:
        if word in punctuation or word in STOP_WORDS or word.strip() == '':
            continue
        if '{NEWLINE}' in word:
            continue
        if word.isnumeric():
            continue
        if word not in keep_words:
            continue
        top_vectors.append(nlp(word).vector)
        top_words.append(word)
        if len(top_vectors) > n_words:
            break

    return top_vectors, top_words

for MODEL in ["full-model", "mighty-nein", "vox-machina"]:
    nlp = spacy.load(f"data/models/crit2vec_{MODEL}")
    vectors = nlp.vocab.vectors.data

#    words = []
#    for i in trange(len(vectors)):
#        words.append(lookup_index(nlp, i))

    data_folder = Path("data")
    with open(data_folder/f"crit2vec_{MODEL}.json", 'r') as f:
        word_info = json.load(f)


#    with open(data_folder/f"{model}-words.json", "w") as f:
#        json.dump(words, f)


    umap_folder = data_folder/MODEL/"umap"
    umap_folder.mkdir(exist_ok=True, parents=True)

    all_n_neighbours = [4, 8, 16, 32, 64]
    all_dims = [2, 3]
    all_n_words = [100, 500, 1000, 5000]

    for dim, n_neighbours, n_words in tqdm(list(product(all_dims, all_n_neighbours, all_n_words))):
        umap = UMAP(n_components=dim, n_neighbors=n_neighbours)
        relevant_vectors, relevant_words = get_relevant_vectors(word_info["words_by_freq"], word_info["keep_words"], nlp, n_words)
        transformed = umap.fit_transform(relevant_vectors)
        np.save(umap_folder/f"neighbours-{n_neighbours}_dim-{dim}_words-{n_words}.npy", transformed)
        with open(umap_folder/f"words-{n_words}.json", "w") as f:
            json.dump(relevant_words, f)
