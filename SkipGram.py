"""
implementing the skip-gram model (with negative sampling).

"""

import pickle
import pandas as pd
import numpy as np
import os, time, re, sys, random, math, collections, nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk import skipgrams
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


# static functions
def who_am_i():
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Natalie Morad', 'id': '000', 'email': 'moradna@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    file = open(fn, "r")
    lines = sent_tokenize(file.read())
    file.close()
    sentences = []
    for line in lines:
        if line == "" or line is None:
            continue
        line = re.sub(r'["|“|”|.|!|?|,]+', "", line)
        line = re.sub(r"'", "", line)
        line = line.lower()
        sentences.append(line)
    return sentences


def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    with open(fn, 'rb') as file:
        sg_model = pickle.load(file)
    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        self.T = None  # Target matrix embeddings
        self.C = None  # Context matrix embeddings
        self.V = None
        # tip 1:
        vocabulary = {}  # {word:count}
        self.vocabulary = {}
        stop_words = set(stopwords.words('english'))

        for line in sentences:
            line_word = line.split()
            for w in line_word:
                if w not in stop_words:
                    if w not in vocabulary:
                        vocabulary[w] = 1
                    else:
                        vocabulary[w] += 1
        for k, v in vocabulary.items():
            if v >= self.word_count_threshold:
                self.vocabulary[k] = v
        self.word_count = self.vocabulary
        self.vocab_size = len(self.vocabulary.keys())
        # tip 2:
        self.word_index = {word: index for index, word in enumerate(self.vocabulary.keys())}
        self.vocabulary = self.word_count
        self.index_word= {index: word for index, word in enumerate(self.vocabulary.keys())}

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default
        if self.V is None:
            return sim
        if w1 not in self.word_index or w2 not in self.word_index:
            return sim  # if word is out of vocabulary, cannot calculate sim

        # Retrieving the vectors for the words
        v1 = self.V[:, self.word_index[w1]]  # get the columns word_index[w1]
        v2 = self.V[:, self.word_index[w2]]  # get the columns word_index[w2]

        # Calculating cosine similarity
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 > 0 and norm_v2 > 0:
            sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        if w != "":
            w = w.lower()
        if w not in self.word_index:
            return []  # if word is not in the vocabulary

        # Initialize a list to hold all words and their similarity to w
        word_sim_lst = []

        for word in self.word_index.keys():
            if word != w:
                # using compute_similarity function for Compute similarity
                sim = self.compute_similarity(w, word)
                word_sim_lst.append((word, sim))

        # Sort by similarity words lst
        word_sim_lst.sort(key=lambda x: x[1], reverse=True)

        return [word for word, sim in word_sim_lst[:n]]  # Return the top n words

    def train(self, T, C, step_size, data_set):
        random.shuffle(data_set)
        epoch_loss = []
        for i, sentance_data_set in enumerate(data_set):
            for t_index, pos_neg in sentance_data_set.items():
                pos_sample, neg_sample = pos_neg[0], pos_neg[1]
                neg_y = np.zeros(len(neg_sample), dtype=int)  # matrix of zero for False label
                pos_y = np.ones(len(pos_sample), dtype=int)  # matrix of once for True label
                y_true = np.concatenate((pos_y, neg_y)).reshape((-1, 1))
                samples_neg_pos = pos_sample + neg_sample

                # Forward step:

                target_embedding = T[:, t_index][:, None]

                pos_neg_matrix = C[samples_neg_pos]  # Calculate the hidden layer
                output_layer = np.dot(pos_neg_matrix, target_embedding)  # Calculate the output layer
                y_pred = sigmoid(output_layer)  # Apply the sigmoid function to get the predicted probabilities
                loss = -np.mean(y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)) + (1 - y_true) * np.log(
                    1 - np.clip(y_pred, 1e-7, 1 - 1e-7)))
                epoch_loss.append(loss)
                error = y_pred - y_true

                # gradient for target embeddings
                T_grad = np.dot(error.T, pos_neg_matrix).T / len(pos_sample + neg_sample)

                # gradient for context embeddings
                C_grad = np.dot(target_embedding, error.T).T
                # update
                pos_neg_matrix -= step_size * C_grad
                C[samples_neg_pos] = pos_neg_matrix
                T[:, [t_index]] -= step_size * T_grad

        return T, C, np.mean(epoch_loss)

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        vocab_size = self.vocab_size
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        delta = 0.001  # if abs(loss i - loss i-1) < delta -> early stop
        Flag_print = False

        pos_context_neg = []  # list of dict, every sentence is a dict: [{word_target : (context samples, negative
        # samples)}, ]
        for sentence in self.sentences:
            sentence = sentence.split()  # split the sentence to tokens
            sentence = [word for word in sentence if
                        word in self.vocabulary]  # check id all the tokens in the vocabulary

            # make positive sample for every target (context)
            pos = list(skipgrams(sentence, 2, self.context // 2 - 1)) + list(
                skipgrams(sentence[::-1], 2, self.context // 2 - 1))

            target_context_dict = {}
            for target, context in pos:
                target_context_dict.setdefault(self.word_index[target], []).append(self.word_index[context])
            if len(target_context_dict.keys()) > 0:
                pos_context_neg.append(target_context_dict)

        for dic in pos_context_neg:
            for target_ind, context_ind in dic.items():
                negative_sample_random = random.choices(list(self.vocabulary.keys()),
                                                        weights=list(self.vocabulary.values()),
                                                        k=self.neg_samples * len(context_ind))
                negative_sample_random = [self.word_index[c] for c in negative_sample_random if c != target_ind]
                dic[target_ind] = (context_ind, negative_sample_random)

        if Flag_print:
            print("the preprocessing finish")
        loss_lst = []  # List to store loss for each epoch
        early_stop_count = 0
        for i in range(1, epochs + 1):
            T, C, avg_loss_epoch = self.train(T, C, step_size,
                                              pos_context_neg)  # pos_context_neg=[{target: ([context, negative_sample])}]

            # update the T and C metrics embeddings
            self.T = T
            self.C = C

            loss_lst.append(avg_loss_epoch)
            if Flag_print:
                print(f"Epoch {i}/{epochs} - Loss: {avg_loss_epoch}")

            # After each epoch check the early stopping: stop training if the Loss was not improved
            if len(loss_lst) > 1:
                if abs(loss_lst[-1] - loss_lst[-2]) < delta:
                    early_stop_count += 1
                    if early_stop_count >= early_stopping:
                        if Flag_print:
                            print(f"Early Stopping")
                        break
                else:
                    early_stop_count = 0

        # save the model:
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        self.V = self.combine_vectors(T, C)
        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        if combo == 0:
            V = T
        elif combo == 1:
            V = C.T
        elif combo == 2:
            V = (C.T + T) / 2
        elif combo == 3:
            V = C.T + T
        elif combo == 4:
            V = np.concatenate((T.T, C), axis=1).T
        else:
            raise ValueError("Invalid value: 'combo' should be number in [0,1,2,3,4].")
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(V, f)
        self.V = V
        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index:
            return ""
        w1, w2, w3 = w1.lower(), w2.lower(), w3.lower()
        v1 = self.V[:, self.word_index[w1]]
        v2 = self.V[:, self.word_index[w2]]
        v3 = self.V[:, self.word_index[w3]]

        v4 = v1 - v2 + v3
        # similarity for each embedding in T
        epsilon = 1e-8  # dived in zero
        sim_embedding = np.dot(self.V.T, v4) / (np.linalg.norm(self.V.T, axis=1) * np.linalg.norm(v4) + epsilon)
        Sorted_embedding = np.argsort(sim_embedding)[::-1]  # Sort by cosine similarity key
        top_5_word = [self.index_word[index] for index in Sorted_embedding][:5]  # closest word index


        for word in top_5_word:
            if word not in {w1, w2, w3}:
                return word

        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        t = self.find_analogy(w1, w2, w3)
        top_n_words = self.get_closest_words(t, n=n)

        if w4 in top_n_words or t == w4:
            return True
        return False
