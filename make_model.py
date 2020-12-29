import json
from pathlib import Path

import spacy
from tqdm import tqdm

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

nlp = spacy.load("en_core_web_md")
merge_ents = nlp.create_pipe("merge_entities")
nlp.add_pipe(merge_ents)

sentences = []
keep_words = set()
folder_name = "data/transcripts"

model = 'full-model'
files = {
    'vox-machina': sorted(Path(folder_name).glob('C1E[0-9][0-9][0-9]*.txt')),
    'mighty-nein': sorted(Path(folder_name).glob('C2E[0-9][0-9][0-9]*.txt')),
    'full-model': sorted(Path(folder_name).glob('C[1-2]E[0-9][0-9][0-9]*.txt'))+sorted(Path(folder_name).glob('CR-*.txt')),
}
transcripts = []
for model, relevant_files in files.items():
    for transcript_file in tqdm(relevant_files):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        transcript = transcript.replace("\n", " \n")
        transcript_doc = nlp(transcript)
        for sentence in transcript_doc.sents:
            sentence_list = [None]*len(sentence)
            for i, token in enumerate(sentence):
                if token.pos_ in {"VERB", "NOUN", "PROPN", "ADJ"}:
                    keep_words.add(token.text.lower().replace("\n", "{NEWLINE}").replace(" ", "_").replace("--", ""))

                sentence_list[i] = token.text.lower().replace("\n", "{NEWLINE}").replace(" ", "_").replace("--", "")

            sentences.append(sentence_list)

    word2vec = Word2Vec(sentences)
    word2vec.wv.save_word2vec_format(f"./data/crit2vec_{model}.txt", binary=False)
    with open(f"data/crit2vec_{model}.words.json", "w") as f:
        json.dump({
            'words_by_freq': word2vec.wv.index2word,
            'keep_words': list(keep_words)
        }, f)
