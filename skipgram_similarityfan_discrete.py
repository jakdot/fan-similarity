"""
Calculating activation based on discretized word-vector values for fan experiments: experiment 2.
"""

import os
from collections import Counter

import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

from skipgram import SkipGramNeg, NegativeSamplingLoss

#specify whether fan for location manipulation should be modeled (using stimuli sets in which location fan is manipulated are used) or person fan shuld be modeled
LOCATION_FAN = True
#LOCATION_FAN = False # this will calculate person fan

def prepare_stim_embeddings(original_list, word_vectors):
    
    # we create a stimuli_list out of original_list (only caring about Person, Location, Article (het/de) belonging to the location;
    if LOCATION_FAN:
        stimuli_list = [f"{s.split()[1]} {s.split()[-1]} {s.split()[-2]}" for s in original_list if len(s.split()) >= 3]
    else:
        stimuli_list = [f"{s.split()[-1]} {s.split()[1]} {s.split()[0].lower()}" for s in original_list if len(s.split()) >= 3]

    # we store words used for retrieval and create lookup tables; we also recreate the text using integers from the lookup tables
    if LOCATION_FAN:
        words = [s.split()[-1] for s in original_list if len(s.split()) >= 3]
    else:
        words = [s.split()[1] for s in original_list if len(s.split()) >= 3]
    vocab_to_int, int_to_vocab = create_lookup_tables(words)

    # in_embeddings are discretized word2vec embeddings: they store 1 for the first 100 elements when that element was greater than 0, and 1 for the following 100 elements when that element was smaller than 0.
    in_embeddings = np.array([word_vectors[int_to_vocab[idx]] for idx in int_to_vocab], dtype=np.float64)

    in_embeddings1 = (in_embeddings > 0).astype(int)
    
    in_embeddings2 = (in_embeddings < 0).astype(int)

    merged = np.hstack((in_embeddings1, in_embeddings2))

    return merged, stimuli_list, vocab_to_int, int_to_vocab

def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

def check_embeddings(embeddings, stimuli_list, vocab_to_int, fan2_list, fan4_list):

    # words_dict is the dict in which key: what is the manipulated fan (so, location if location is manipulated fan); value: the other element (person, if location is the key)
    words_dict = {x.split()[1]: x.split()[0] for x in stimuli_list}

    # create fan dict (assembling any fan)
    fan = dict()

    for stimulus in words_dict:
        word_vec = embeddings[vocab_to_int[stimulus],]
        # calculating fan:
        # diagonalize the vector
        diag_vec = np.diag(word_vec)
        # matmul -> matrix in which only embeddings that have 1 at the same position as word_vec stay
        merge = np.matmul(embeddings, diag_vec)
        # sum over columns
        summed_fans = np.sum(merge, axis=0)
        # calculate fan: log(50) represents the size of the whole vocabulary (there are 50 words)
        fan[stimulus] = np.log(50) - np.log(summed_fans[summed_fans != 0])

    # create fan dicts (for fan2 and fan4)
    if LOCATION_FAN:
        fan2 = {s.split()[-1]: np.nan for s in fan2_list if len(s.split()) >= 3}
        fan4 = {s.split()[-1]: np.nan for s in fan4_list if len(s.split()) >= 3}
    else:
        fan2 = {s.split()[1]: np.nan for s in fan2_list if len(s.split()) >= 3}
        fan4 = {s.split()[1]: np.nan for s in fan4_list if len(s.split()) >= 3}


    for x in fan:
        if x in fan2:
            #print("Fan2")
            #print(x)
            #print(fan[x])
            fan2[x] = fan[x].mean()
            #print(fan2[x])
        elif x in fan4:
            #print("Fan4")
            #print(x)
            #print(fan[x])
            fan4[x] = fan[x].mean()
            #print(fan4[x])
        else:
            raise Exception("Wrong number of fan")

    return np.array(list(fan2.values())).mean(), np.array(list(fan4.values())).mean()

def calculate_activation(lists_checked):
    """
    Calculate activation (=pmi).
    :lists_checked list of numbers specifying what stimuli list should be checked
    """

    # these lists will store average activations for each list
    fan2_final = []
    fan4_final = []

    # check first the fan without any training
    for i in lists_checked:

        # fan2 and 4 for each case (=stimuli list)
        fan2_case = []
        fan4_case = []

        testing_selected = testing[(testing["pp_num"] == i) & (testing["condition"] == "target")]

        if LOCATION_FAN:
            fan2_list = testing_selected[testing_selected["fan_loc"] == 2]["test_sentence"].drop_duplicates().to_list()
            #print(fan2_list)
            fan4_list = testing_selected[testing_selected["fan_loc"] == 4]["test_sentence"].drop_duplicates().to_list()
            #print(fan4_list)
        else:
            fan2_list = testing_selected[testing_selected["fan_pers"] == 2]["test_sentence"].drop_duplicates().to_list()
            #print(fan2_list)
            fan4_list = testing_selected[testing_selected["fan_pers"] == 4]["test_sentence"].drop_duplicates().to_list()
            #print(fan4_list)

        original_list = testing_selected["test_sentence"].drop_duplicates().to_list()
    
        in_embeddings, stimuli_list, vocab_to_int, int_to_vocab = prepare_stim_embeddings(original_list, word_vectors)
        fan2, fan4 = check_embeddings(in_embeddings, stimuli_list, vocab_to_int, fan2_list, fan4_list)
        fan2_case.append(fan2)
        fan4_case.append(fan4)
    
        fan2_final.append(sum(fan2_case)/len(fan2_case))
        fan4_final.append(sum(fan4_case)/len(fan4_case))

    return fan2_final, fan4_final

def store_results(lists_checked, fan2_final, fan4_final, csv_filename, droprate=0, training="False"):

    df = pd.DataFrame({'Lists': lists_checked + ["average"], 'Fan2': fan2_final, 'Fan4': fan4_final, 'Fan2greater': [fan2_final[i] > fan4_final[i] for i in range(len(fan2_final))], 'Drop': [droprate for _ in range(len(fan2_final))], 'Training': [training for _ in range(len(fan2_final))]})

    file_exists = os.path.exists(csv_filename)

    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

# load stimuli data
testing = pd.read_csv("online_testing_all.csv")

# load vectors
# the model downloaded from https://aclanthology.org/W17-0237 (Dutch CoNLL17 corpus, skipgram)
word_vectors = KeyedVectors.load_word2vec_format("39/model.bin", binary=True, limit=700000)  # Loads only top 700,000 words (covers everything in the fan experiment)

# select stimuli lists that manipulate loc
if LOCATION_FAN:
    lists_checked = [i for i in range(1, 60, 2)]
    csv_filename = "discrete_similarityfan_loc.csv"
else:
    lists_checked = [i for i in range(2, 61, 2)]
    csv_filename = "discrete_similarityfan_pers.csv"

# estimate activation without any training
fan2_final, fan4_final = calculate_activation(lists_checked)

fan2_final.append(sum(fan2_final)/len(fan2_final))
fan4_final.append(sum(fan4_final)/len(fan4_final))

store_results(lists_checked, fan2_final, fan4_final, csv_filename)
