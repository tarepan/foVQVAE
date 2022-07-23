# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
#
#
import sys
import glob
import pickle
import os
import multiprocessing as mp

from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId # pyright: ignore [reportMissingTypeStubs]
from speechcorpusy import load_preset                                     # pyright: ignore [reportMissingTypeStubs]; bacause of library

from utils.dsp import *

CORPUS_NAME = sys.argv[1]
DATA_PATH = sys.argv[2]

def get_files(corpus_name: str):

    corpus = load_preset(corpus_name, root=".")
    corpus.get_contents()

    all_utterances = corpus.get_identities()
    spks = []
    for spk in sorted(set(map(lambda item_id: item_id.speaker, all_utterances))):
        ids_spk_x = filter(lambda item_id: item_id.speaker == spk, all_utterances)
        path_name_x = list(map(lambda item_id: (str(corpus.get_item_path(item_id)), item_id.name), ids_spk_x))
        spks.append(path_name_x)

    return spks

path_name_spks = get_files(CORPUS_NAME)


def process_file(i: int, i_path: str, name: str):
    dir = f'{DATA_PATH}/{i}'
    o_filename = f'{dir}/{name}.npy'
    if os.path.exists(o_filename):
        print(f'{o_filename} already exists, skipping')
        return

    # Load
    floats = load_wav(i_path, encode=False)

    # Silent Trimming
    trimmed, _ = librosa.effects.trim(floats, top_db=25)

    # float->int
    quant = (trimmed * (2**15 - 0.5) - 0.5).astype(np.int16)

    # Length check
    if max(abs(quant)) < 2048:
        print(f'audio fragment too quiet ({max(abs(quant))}), skipping: {i_path}')
        return
    if len(quant) < 10000:
        print(f'audio fragment too short ({len(quant)} samples), skipping: {i_path}')
        return

    os.makedirs(dir, exist_ok=True)
    np.save(o_filename, quant)
    return name


index = []
for i, spk_x in enumerate(path_name_spks):
    res = [process_file(i, path, name) for (path, name) in spk_x]
    index.append([x for x in res if x])
    print(f'Done processing speaker {i}')

os.makedirs(DATA_PATH, exist_ok=True)
with open(f'{DATA_PATH}/index.pkl', 'wb') as f:
    pickle.dump(index, f)
