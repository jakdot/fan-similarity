# README

This folder checks binary attribute extractions, following the idea of Korchinski et al. On the emergence of linear analogies in word embeddings. After it works we can put them in an ACT-R model.

We need to get trarget and context vectors, so we first collect those from Dutch wikipedia, using gensim word2vec skipgram model. Then, we caluclate PMI and attributes. Finally, we take attributes and use them in skipgram_similarityfan_discrete.py. This gives us actr_activations_perword.csv, and discrete_similarityfan_loc.csv and discrete_similarityfan_pers.csv.

The process is:

prepare_dutch_wkipedia -> train_on_dutch_wiki -> sanity_check_and_extract_attributes -> skipgram_similarityfan_discrete

The result is a bit better than when we binarize target vectors directly, but not by much. This is not a problem of our word2vec skipgram model: when we check it on skipgram_similarityfan, we get very similar results to the ones from the pre-trained word2vec model.

