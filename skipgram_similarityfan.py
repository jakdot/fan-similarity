"""
Training and testing skipgram for fan experiments: experiment 2.
"""

import random
import os
from collections import Counter

import torch
from torch import optim
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

from skipgram import SkipGramNeg, NegativeSamplingLoss

#specify whether fan for location manipulation should be modeled (using stimuli sets in which location fan is manipulated are used) or person fan shuld be modeled
LOCATION_FAN = True
#LOCATION_FAN = False # this will calculate person fan

def train_skipgram(n_epochs, embedding_dim, original_list, word_vectors, droprate):
    
    # we create a stimuli_list out of original_list (only caring about Person, Location, Article (het/de) belonging to the location; we then join the result into one text
    if LOCATION_FAN:
        stimuli_list = [f"{s.split()[1]} {s.split()[-1]} {s.split()[-2]}" for s in original_list if len(s.split()) >= 3]
    else:
        stimuli_list = [f"{s.split()[-1]} {s.split()[1]} {s.split()[0].lower()}" for s in original_list if len(s.split()) >= 3]

    text = " ".join(stimuli_list)
    
    # we split text into words and create lookup tables; we also recreate the text using integers from the lookup tables
    words = text.split()
    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    in_embeddings = torch.tensor([word_vectors[int_to_vocab[idx]] for idx in int_to_vocab], dtype=torch.float)

    print(in_embeddings.shape)

    #out_embeddings = torch.transpose(torch.tensor([ft.get_word_vector(int_to_vocab[idx]) for idx in int_to_vocab], dtype=torch.float), 0, 1)
    #out_embeddings = torch.tensor([word_vectors[int_to_vocab[idx]] for idx in int_to_vocab], dtype=torch.float)

    #print(out_embeddings.shape)

    # we specify noise_dist (if needed) and initialize a model, loss function and optimizer
    noise_dist = compute_noise_distribution(int_words, vocab_to_int)
    model = SkipGramNeg(len(vocab_to_int), embedding_dim, in_embeddings, None, noise_dist)
    criterion = NegativeSamplingLoss(droprate)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # we train model and return the model, stimuli list and lookup tables
    train_model(model, criterion, optimizer, int_words, int_to_vocab, n_epochs)
    return model, stimuli_list, vocab_to_int, int_to_vocab

def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

def compute_noise_distribution(int_words, vocab_to_int):
    freq = Counter(int_words)
    freq_ratio = np.array([freq[vocab_to_int[word]] / len(vocab_to_int) for word in vocab_to_int])
    unigram_dist = freq_ratio / freq_ratio.sum()
    # noise dist is based on the original word2vec: prob^{3/4}, normalized
    return torch.from_numpy(unigram_dist**0.75 / np.sum(unigram_dist**0.75))

def get_target(words, idx, max_window_size=1):
    # randomly select target from the context around the word
    # note that we are using window size 1 so there is no randomness involved
    R = random.randint(1, max_window_size)
    start = max(0,idx-R)
    end = min(idx+R,len(words)-1)
    targets = words[start:idx] + words[idx+1:end+1] # +1 since doesn't include this idx
    return targets

def get_batches(words, batch_size, max_window_size=1):
    # create batches
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size] #incomplete batch is thrown away
    # this does not matter for us, since we match batches to length
    for i in range(0, len(words), batch_size):
        batch_of_center_words = words[i:i+batch_size]   # current batch of words
        batch_x, batch_y = [], []  

        for ii in range(len(batch_of_center_words)):  # range(batch_size) unless truncated at the end
            x = [batch_of_center_words[ii]]             # single word
            y = get_target(words=batch_of_center_words, idx=ii, max_window_size=max_window_size)  # list of context words

            batch_x.extend(x * len(y)) # repeat the center word (n_context_words) times
            batch_y.extend(y)

        # if batch_size is 3, then:
        # batch_x: ["boer", "zwembad", "zwembad", "het"]
        # batch_y: ["zwembad", "boer", "het", "zwembad"]
        # so, x is input, y is target (context word, next to input word)
        yield batch_x, batch_y

def train_model(model, criterion, optimizer, int_words, int_to_vocab, n_epochs):
    for epoch in range(n_epochs):
        for inputs, targets in get_batches(int_words, batch_size=3):
            inputs, targets = torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
            loss = criterion(model.forward_input(inputs), model.forward_target(targets), model.forward_noise(len(inputs)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f}")

def check_embeddings(model, stimuli_list, vocab_to_int, fan2_list, fan4_list):

    # embeddings are used for in_weights (input words)
    embeddings = model.in_embed.weight.to('cpu').data.numpy()
    # contexts are predicted words
    contexts = model.out_embed.weight.to('cpu').data.numpy()

    # words_dict is the dict in which key: what is the manipulated fan (so, location if location is manipulated fan); value: the other element (person, if location is the key)
    words_dict = {x.split()[1]: x.split()[0] for x in stimuli_list}

    # complete_check checks dot product for every word of every stimulus
    complete_check = dict()

    for stimulus in words_dict:

        word_vec = embeddings[vocab_to_int[stimulus]]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))

        complete_check[stimulus] = {}
        for j in words_dict:
            # calculate dot product
            complete_check[stimulus].update({words_dict[j]: float(np.matmul(contexts[vocab_to_int[words_dict[j]]], word_vec).flatten())})

    for x in complete_check:
        # we check if the top element is the word that should be recalled
        if not words_dict[x] == list(zip(*sorted(complete_check[x].items(), key=lambda item: item[1], reverse=True)))[0][0]:
            print("Failure for recall for", x)
            #return (np.nan, np.nan) # uncomment if you want to stop checking whenever there is a mistake in recall

    # create fan dict (assembling any fan)
    fan = dict()

    for stimulus in words_dict:
        word_vec = embeddings[vocab_to_int[stimulus]]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))
        fan[stimulus] = float(np.matmul(contexts[vocab_to_int[words_dict[stimulus]]], word_vec).flatten())

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
            fan2[x] = fan[x]
            #print(fan2[x])
        elif x in fan4:
            #print("Fan4")
            #print(x)
            #print(fan[x])
            fan4[x] = fan[x]
            #print(fan4[x])
        else:
            raise Exception("Wrong number of fan")

    return np.array(list(fan2.values())).mean(), np.array(list(fan4.values())).mean()

def estimate_activation(lists_checked, epochs, droprate=0.0):
    """
    Empirically estimate activation (=pmi).
    :lists_checked list of numbers specifying what stimuli list should be checked
    :epochs list of number of epochs used in training
    :droprate what droprate in the model training should be used
    """

    # these lists will store average activations for each list
    fan2_final = []
    fan4_final = []

    # for each list, calcualte fan2 and fan4 activations
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
    
        for myseed in [23, 40, 50]:

            torch.manual_seed(myseed)
            torch.cuda.manual_seed(myseed)
            random.seed(myseed)
            
            for n_epochs in epochs:

                dim = 100
                model, stimuli_list, vocab_to_int, int_to_vocab = train_skipgram(n_epochs, dim, original_list, word_vectors, droprate)
                fan2, fan4 = check_embeddings(model, stimuli_list, vocab_to_int, fan2_list, fan4_list)
                fan2_case.append(fan2)
                fan4_case.append(fan4)
    
        fan2_final.append(sum(fan2_case)/len(fan2_case) + np.log(5))
        fan4_final.append(sum(fan4_case)/len(fan4_case) + np.log(5))

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

# select stimuli lists that manipulate loc or pers
if LOCATION_FAN:
    lists_checked = [i for i in range(1, 60, 2)]
    csv_filename = "similarityfan_loc.csv"
else:
    lists_checked = [i for i in range(2, 61, 2)]
    csv_filename = "similarityfan_pers.csv"

# estimate activation without any training
fan2_final, fan4_final = estimate_activation(lists_checked, epochs=[0])

fan2_final.append(sum(fan2_final)/len(fan2_final))
fan4_final.append(sum(fan4_final)/len(fan4_final))

store_results(lists_checked, fan2_final, fan4_final, csv_filename)

# estimate activation with training, and for different droprates and epochs
for droprate in [0.0, 0.2, 0.4]:

    fan2_final, fan4_final = estimate_activation(lists_checked, epochs=[30, 50, 70, 100], droprate=droprate)

    print(fan2_final)
    print(fan4_final)

    fan2_final.append(sum(fan2_final)/len(fan2_final))
    fan4_final.append(sum(fan4_final)/len(fan4_final))

    store_results(lists_checked, fan2_final, fan4_final, csv_filename, droprate=droprate, training="True")
