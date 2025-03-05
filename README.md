# fan-similarity
Models testing fan effects based on similarity

The files test whether skipgram models can predict fan effects for simple fan experiments (as in Anderson) and for fan effects based on similarity.

Code:

- skipgram.py: skipgram model in pytorch
- skipgram_fan.py: code to model Anderson's fan experiment; it assumes no pretrained data, just learns on Anderson's experiment; the main output is activation due to the skipgram model; this matches the theoretical value, pmi (which would be the ACT-R value for spreading activation).
- skipgram_similarityfan.py: code to model the fan similarity experiment; it uses pretrained embeddings for words, and learns contexts, which are initialized randomly; it measures the difference in activation between fan2 and fan4 for 30 files manipulating location fan and 30 files manipulating person fan.
- skipgram_similarityfan_discrete.py: code to model the fan similarity experiment; it uses pretrained embeddings for words, but discretizes them using the sign(-1, +1); then, it applies the standard ACT-R formula, treating dimensions in vector as cues (contexts).

Outputs:
- output_skipgram_fan.txt: the output of skipgram_fan.py
- similarityfan_loc.csv: the csv output of skipgram_similarityfan.py for location fan stimuli
- similarityfan_pers.csv: the csv output of skipgram_similarityfan.py for person fan stimuli
- discrete_similarityfan_loc.csv: the csv output of skipgram_similarityfan_discrete.py for location fan stimuli
- discrete_similarityfan_pers.csv: the csv output of skipgram_similarityfa_discreten.py for person fan stimuli

Others:
- info_possible_models.csv: info on models that we considered; CoNLL2017 was used for the simulations here
- online_testing_all.csv: stimuli for the similarity fan experiment
