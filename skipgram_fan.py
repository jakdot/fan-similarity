"""
Training and testing skipgram for fan experiments: experiment 1.
"""

import random
from collections import Counter

import torch
from torch import optim
import numpy as np

from skipgram import SkipGramNeg, NegativeSamplingLoss

def train_skipgram(n_epochs, embedding_dim):
    # original_list consists 24 elements, replicating one experimental list
    # in the list, some words appear twice (fan=2), some words appear 4 times (fan=4)
    original_list = [
        "De boer is in het zwembad", "De boer is in de kroeg", "De leraar is in het zwembad", "De leraar is in de winkel",
        "De schilder is in de kroeg", "De schilder is in het park", "De kapper is in de winkel", "De kapper is in het park",
        "De soldaat is in het dorp", "De soldaat is in de bioscoop", "De artiest is in het dorp", "De artiest is in de bioscoop",
        "De rechter is in het dorp", "De rechter is in het huis", "De schrijver is in het dorp", "De schrijver is in het huis",
        "De zeeman is in de bioscoop", "De zeeman is in het ravijn", "De kapitein is in de bioscoop", "De kapitein is in het ravijn",
        "De advocaat is in het huis", "De advocaat is in het ravijn", "De ober is in het huis", "De ober is in het ravijn"
    ]

    # we create a stimuli_list out of it (only caring about Person, Location, Article (het/de) belonging to the location; we then join the result into one text
    stimuli_list = [f"{s.split()[1]} {s.split()[-1]} {s.split()[-2]}" for s in original_list if len(s.split()) >= 3]
    text = " ".join(stimuli_list)
    
    # we split text into words and create lookup tables; we also recreate the text using integers from the lookup tables
    words = text.split()
    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]
    
    # we specify noise_dist (if needed) and initialize a model, loss function and optimizer
    noise_dist = compute_noise_distribution(int_words, vocab_to_int)
    model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist) #noise_dist if we want to match word2vec closer; None if we want to have uniform
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # we train model and return the model, stimuli list and lookup tables
    train_model(model, criterion, optimizer, int_words, n_epochs)
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

def train_model(model, criterion, optimizer, int_words, n_epochs):
    for epoch in range(n_epochs):
        for inputs, targets in get_batches(int_words, batch_size=3):
            inputs, targets = torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
            loss = criterion(model.forward_input(inputs), model.forward_target(targets), model.forward_noise(len(inputs)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f}")

def check_embeddings(model, stimuli_list, vocab_to_int):
    """
    Check whether (trained) model leads to selecting the correct elements. If not, print which word leads to selecting the incorrect context word.
    Store average activations for fan2 and fan4 afterwards.
    """
    # embeddings are used for in_weights (input words)
    embeddings = model.in_embed.weight.to('cpu').data.numpy()
    # contexts are predicted words
    contexts = model.out_embed.weight.to('cpu').data.numpy()

    # words_dict is the dict in which key: what is the manipulated fan (so, location if location is manipulated fan); value: the other element (person, if location is the key)
    words_dict = {x.split()[1]: set() for x in stimuli_list}
    for x in stimuli_list:
        words_dict[x.split()[1]].add(x.split()[0])

    # complete_check checks dot product for every word of every stimulus
    complete_check = dict()

    for stimulus in words_dict:

        word_vec = embeddings[vocab_to_int[stimulus]]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))

        complete_check[stimulus] = {}
        for j in words_dict:
            for k in words_dict[j]:
                # calculate dot product
                complete_check[stimulus].update({k: float(np.matmul(contexts[vocab_to_int[k]], word_vec).flatten())})

    for x in complete_check:
        # we check if the top n(=2,4) words are the words that should be recalled
        if not set(words_dict[x]) == set(list(zip(*sorted(complete_check[x].items(), key=lambda item: item[1], reverse=True)))[0][:len(words_dict[x])]):
            print("Failure for recall for", x)
            #return (np.nan, np.nan) # uncomment if you want to stop checking whenever there is a mistake in recall
    
    # we now only care about the actual fan - so the context words that do match the recall
    fan = dict()

    for stimulus in words_dict:

        word_vec = embeddings[vocab_to_int[stimulus]]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))

        fan[stimulus] = {}

        for j in words_dict[stimulus]:
            # calculate dot product
            fan[stimulus].update({j: float(np.matmul(contexts[vocab_to_int[j]], word_vec).flatten())})

    # split fan into fan2 dict and fan4 dict
    fan2 = dict()
    fan4 = dict()

    for x in fan:
        if len(words_dict[x]) == 2:
            #print("Fan2")
            #print(x)
            #print(fan[x])
            fan2[x] = np.array(list(fan[x].values())).mean()
            #print(fan2[x])
        elif len(words_dict[x]) == 4:
            #print("Fan4")
            #print(x)
            #print(fan[x])
            fan4[x] = np.array(list(fan[x].values())).mean()
            #print(fan4[x])
        else:
            raise Exception("Wrong number of fan")

    return np.array(list(fan2.values())).mean(), np.array(list(fan4.values())).mean()

def estimate_activation(dims, epochs):
    """
    Empirically estimate activation (=pmi).
    :dims list of vector dimensions
    :epochs list of number of epochs used in training
    """
    stored_fan2 = {x: [] for x in dims}
    stored_fan4 = {x: [] for x in dims}
    for dim in stored_fan2:
        for n_epochs in epochs:
            for seed in [31, 37, 42]:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                random.seed(seed)
                print("N_EPOCH: ", n_epochs)
                print("DIM: ", dim)
                model, stimuli_list, vocab_to_int, int_to_vocab = train_skipgram(n_epochs, dim)
                fan2, fan4 = check_embeddings(model, stimuli_list, vocab_to_int)
                # print estimated fan 2 and 4 per cycle
                print("Fan 2: ", fan2, "Fan 4 :", fan4)
                print("Shifted: Fan 2: ", fan2 + np.log(5), "Fan 4 :", fan4 + np.log(5))
                stored_fan2[dim].append(fan2 + np.log(5))
                stored_fan4[dim].append(fan4 + np.log(5))

    return stored_fan2, stored_fan4

# Calculated pmi
print("Calculated pmi")
print("Fan 2: ", np.log(1 * 24/(2*2)), "Fan 4: ", np.log(1 * 24/(2*4)))
print("================")

stored_fan2, stored_fan4 = estimate_activation([30, 50, 100, 300], [100])

print("++++++++++++++++++++")
print("FAN 2: ", {x: sum(stored_fan2[x])/len(stored_fan2[x]) for x in stored_fan2})
print("FAN 4: ", {x: sum(stored_fan4[x])/len(stored_fan4[x]) for x in stored_fan4})
print("This should match the calculated pmi with around 10% error.")
print("The match is a bit better if you use no noise distribution.")
print("++++++++++++++++++++")

# Some more explorations -- fewer epochs, fewer dims
estimate_activation([5, 10, 50], [10, 20, 30])

