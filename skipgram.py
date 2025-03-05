"""
Skipgram model. Based on: https://github.com/lukysummer/SkipGram_with_NegativeSampling_Pytorch.git
"""

import torch
from torch import nn

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, in_embeddings=None, out_embeddings=None, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # Initialize both embedding tables with uniform distribution, unles they are given
        if in_embeddings != None:
            self.in_embed = nn.Embedding.from_pretrained(in_embeddings, freeze=True)
        else:
            self.in_embed = nn.Embedding(n_vocab, n_embed)
            self.in_embed.weight.data.uniform_(-1, 1)

        if out_embeddings != None:
            self.out_embed = nn.Embedding.from_pretrained(out_embeddings, freeze=False)
        else:
            self.out_embed = nn.Embedding(n_vocab, n_embed)
            self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings

    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings

    def forward_noise(self, batch_size, n_samples=5):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # If no Noise Distribution specified, sample noise words uniformly from vocabulary
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # torch.multinomial :
        # Returns a tensor where each row contains (num_samples) **indices** sampled from 
        # multinomial probability distribution located in the corresponding row of tensor input.
        noise_words = torch.multinomial(input       = noise_dist,           # input tensor containing probabilities
                                        num_samples = batch_size*n_samples, # number of samples to draw
                                        replacement = True)
        noise_words = noise_words.to("cpu")
        
        # use context matrix for embedding noise samples
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors


class NegativeSamplingLoss(nn.Module):
    def __init__(self, droprate=0):
        super().__init__()
        self.droprate = droprate

    def forward(self, 
              input_vectors, 
              output_vectors, 
              noise_vectors):
      
        batch_size, embed_size = input_vectors.shape
    
        input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors

        drop = nn.Dropout(p=self.droprate)

        input_vectors = drop(input_vectors)
        output_vectors = drop(output_vectors)

        # log-sigmoid loss for correct pairs
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()

        # log-sigmoid loss for incorrect pairs
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        return -(out_loss + noise_loss).mean()  # average batch loss

