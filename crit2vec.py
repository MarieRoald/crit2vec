import itertools
import operator

import numpy as np
import spacy


def lookup_index(nlp, index):
    """Get the words associated with the vector of given index in the model.
    """
    words = []
    for key, row in nlp.vocab.vectors.key2row.items():
        if row == index:
            words.append(nlp.vocab.strings[key])
    return words


def lookup_n_nearest(nlp, word_vector, n):
    """Find the words associated with the n nearest word vectors in the corpus.
    """
    normalised_vectors = nlp.vocab.vectors.data / np.linalg.norm(nlp.vocab.vectors.data, axis=1, keepdims=True)
    similarity_scores = normalised_vectors @ word_vector / np.linalg.norm(word_vector)
    return [
        (lookup_index(nlp, idx), similarity)
        for idx, similarity in sorted(enumerate(similarity_scores), key=lambda x: -x[1])[:n]
    ]


def add_words(nlp, words, use_norm=True):
    """Add the vectors corresponding to the given words, potentially after normalising them.
    """
    vector = 0
    for word in words:
        current_vector = nlp(word.lower()).vector
        if use_norm:
            current_vector = current_vector/np.linalg.norm(current_vector)
        vector += current_vector
    return vector


def word_arithmetic(nlp, positive=None, negative=None, use_norm=True):
    if positive is None:
        positive = []
    if negative is None:
        negative = []

    vector = add_words(nlp, positive, use_norm=use_norm)
    vector -= add_words(nlp, negative, use_norm=use_norm)

    return vector


def find_nearest(nlp, positive=None, negative=None, n_vectors=5, use_norm=True,):
    """Find the `n_vectors` nearest word to an arithmetic expression, disregarding all words in the expression.
    """
    if positive is None:
        positive = []
    if negative is None:
        negative = []
    input_words = set(positive) | set(negative)

    vector = word_arithmetic(nlp, positive=positive, negative=negative, use_norm=use_norm)
    nearest_words, similarities = _unzip_two(lookup_n_nearest(nlp, vector, n_vectors + len(input_words)))
    
    # Add `n_words` words that are not included in the input-expression
    nearest_word_dict = {}
    exit_loop = False
    for words, similarity in zip(nearest_words, similarities):
        for word in words:
            if word in input_words:
                continue
            nearest_word_dict[word] = similarity
            if len(nearest_word_dict) >= n_vectors:
                exit_loop = True
                break
        if exit_loop:
            break
        
    return nearest_word_dict


def _unzip_two(merged):
    it1, it2 = itertools.tee(merged)
    unmerged_1 = map(operator.itemgetter(0), it1)
    unmerged_2 = map(operator.itemgetter(1), it2)
    return unmerged_1, unmerged_2


if __name__ == "__main__":
    NLP = spacy.load("models/crit2vec_full-model")
    NORMALISED_VECTORS = NLP.vocab.vectors.data / np.linalg.norm(NLP.vocab.vectors.data)

    print(find_nearest(NLP, positive=["vax", "sister"], negative=["brother"], use_norm=False))
    print(find_nearest(NLP, positive=["raishan"],))
