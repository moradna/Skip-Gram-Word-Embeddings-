import itertools
import os
import unittest
from collections import Counter

import numpy as np
from SkipGram  import SkipGram, normalize_text, sigmoid

fn = 'corpus.txt'
file_path = "model.pkl"
sentences = normalize_text(fn)


class TestSkipGram(unittest.TestCase):

    def setUp(self):
        self.sentences = sentences
        self.d = 100
        self.neg_samples = 4
        self.context = 4
        self.word_count_threshold = 1
        self.word_count = Counter(itertools.chain(*self.sentences))
        self.word_index = {word: i for i, word in enumerate(self.word_count.keys())}
        self.index_word = {i: word for word, i in self.word_index.items()}
        self.skipgram = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        self.skipgram.learn_embeddings()

    def tearDown(self):
        self.skipgram = None

    # test_sigmoid
    ####################################################################################################################
    def test_sigmoid(self):
        x = 0.5
        expected = 0.6224593312018546
        result = sigmoid(x)
        self.assertEqual(result, expected)

    # normalize_text
    ####################################################################################################################
    def test_normalize_text(self):
        expected = [['hey', 'dont', 'like', 'way', 'im', 'talking'], ['hey', 'stand', 'keep', 'call', 'names'],
                    ['im', 'enemy'], ['youre', 'gon', 'na', 'dont']]
        result = normalize_text(fn)[:4]
        self.assertEqual(result, expected)

    # compute_similarity
    ####################################################################################################################
    def test_compute_similarity(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.1], [0.2, 0.1]])
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        similarity = self.skipgram.compute_similarity(w1="unicorn", w2="power")
        expected = 0.9284766908852594
        self.assertEqual(similarity, expected)
#
    def test_compute_similarity_sim_equal_zero(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.0, 0.0], [0.0, 0.0], [0.9, 0.1], [0.1, 0.1], [0.2, 0.1]])
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        similarity = self.skipgram.compute_similarity(w1="unicorn", w2="power")
        expected = 0.0
        self.assertEqual(similarity, expected)

    # get_closest_words
    ####################################################################################################################
    def test_get_closest_words_true_res(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.1], [0.2, 0.1]])
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}
        closest_words = self.skipgram.get_closest_words('unicorn', n=2)
        self.assertEqual(closest_words, ['power', 'back'])

    def test_get_closest_words(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.9, 0.9], [0.9, 0.9]])
        self.skipgram.word_count_threshold = 0
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}
        closest_words = self.skipgram.get_closest_words('learn', n=2)
        self.assertNotEqual(closest_words, ['power', 'back'])

    def test_get_closest_words_empty_list(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.1], [0.2, 0.1]])
        closest_words = self.skipgram.get_closest_words('bla', n=2)
        self.assertEqual(closest_words, [])

    # test_analogy
    ####################################################################################################################
    def test_test_analogy_True_return(self):
        v1 = np.random.rand(4)
        v2 = np.random.rand(4)
        v3 = np.random.rand(4)
        v4 = v1 - v2 + v3
        self.skipgram.V = np.array([v1, v2, v3, v4])

        # Define the word_index dictionary
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}

        analogy_words = self.skipgram.test_analogy('unicorn', 'power', 'back', 'look', n=1)
        self.assertTrue(analogy_words)

    def test_test_analogy_False_return(self):
        v1 = np.random.rand(4)
        v2 = np.random.rand(4)
        v3 = np.random.rand(4)
        v4 = v1 - v2 + v3
        v5 = np.random.rand(4)
        v6 = np.random.rand(4)

        self.skipgram.V = np.array([v1, v2, v3, v4, v5, v6])
        # Define the word_index dictionary
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4, 'turn': 5, 'around': 6}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}
        analogy_words = self.skipgram.test_analogy('unicorn', 'power', 'back', 'around', n=1)
        self.assertFalse(analogy_words)

    def test_test_analogy_empty(self):
        v1 = np.random.rand(4)
        v2 = np.random.rand(4)
        v3 = np.random.rand(4)
        v4 = np.array([0.0, 0.0, 0.0, 0.0])

        self.skipgram.V = np.array([v1, v2, v3, v4])

        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}
        analogy_words = self.skipgram.test_analogy('unicorn', 'power', 'around', 'look', n=1)
        self.assertFalse(analogy_words)

    # find_analogy
    ####################################################################################################################
    def test_find_analogy_empty_list(self):
        v1 = np.random.rand(4)
        v2 = np.random.rand(4)
        v3 = np.random.rand(4)
        v4 = np.random.rand(4)

        self.skipgram.V = np.array([v1, v2, v3, v4])

        # Define the word_index dictionary
        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}

        analogy_words = self.skipgram.find_analogy('unicorn', 'power', 'you')
        self.assertEqual('', analogy_words)

    def test_find_analogy_different_analogy(self):
        v1 = np.random.rand(4)
        v2 = np.random.rand(4)
        v3 = np.random.rand(4)
        v4 = np.random.rand(4)
        v5 = np.random.rand(4)
        v6 = np.array([0.0, 0.0, 0.0, 0.0])

        self.skipgram.V = np.array([v1, v2, v3, v4, v5, v6])

        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4, 'turn': 5, 'around': 6}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}

        analogy_words = self.skipgram.find_analogy('unicorn', 'power', 'look')
        self.assertNotEqual('around', analogy_words)

    def test_find_analogy_return_correct_list(self):
        v1 = np.array([0.3, 0.2, 0.5, .07])
        v2 = np.array([0.1, 0.1, 0.5, 0.4])
        v3 = np.array([0.9, 0.9, 0.2, 0.4])
        v4 = v1 - v2 + v3
        self.skipgram.V = np.array([v1, v2, v3, v4])

        self.skipgram.word_index = {'unicorn': 0, 'power': 1, 'back': 2, 'look': 3, 'learn': 4}
        self.skipgram.index_word = {i: word for word, i in self.skipgram.word_index.items()}

        analogy_words = self.skipgram.find_analogy('unicorn', 'power', 'back')
        self.assertEqual("look", analogy_words)

    # learn_embeddings
    ####################################################################################################################
    def test_learn_embeddings(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        self.assertEqual(C.shape, (len(sg.word_count), sg.d))
        self.assertEqual(T.shape, (sg.d, len(sg.word_count)))

    def test_learn_embeddings_file_generated(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        if os.path.exists(file_path):
            os.remove(file_path)
        sg.learn_embeddings(step_size=0.0001, epochs=10, early_stopping=3, model_path=file_path)
        self.assertTrue(os.path.exists(fn))

    # combine_vectors
    ####################################################################################################################
    def test_combine_vectors_combo_0(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        V = sg.combine_vectors(T, C, combo=0, model_path=None)
        self.assertEqual(V.shape, (len(sg.word_count), sg.d))

    def test_combine_vectors_combo_1(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        V = sg.combine_vectors(T, C, combo=1, model_path=None)
        self.assertEqual(V.shape, (len(sg.word_count), sg.d))

    def test_combine_vectors_combo_2(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        V = sg.combine_vectors(T, C, combo=2, model_path=None)
        self.assertEqual(V.shape, (len(sg.word_count), sg.d))

    def test_combine_vectors_combo_3(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        V = sg.combine_vectors(T, C, combo=3, model_path=None)
        self.assertEqual(V.shape, (len(sg.word_count), sg.d))

    def test_combine_vectors_combo_4(self):
        sg = SkipGram(self.sentences, self.d, self.neg_samples, self.context, self.word_count_threshold)
        T, C = sg.learn_embeddings(step_size=0.001, epochs=10, early_stopping=3, model_path=None)
        V = sg.combine_vectors(T, C, combo=4, model_path=None)
        self.assertEqual(V.shape, (len(sg.word_count), sg.d + sg.d))

    # _compute_similarity
    # ####################################################################################################################
    def test_compute_similarity(self):
        # Set up the SkipGram object with a sample embedding matrix
        self.skipgram.V = np.array([[0.5, 0.5], [0.7, 0.3]])
        similarity = self.skipgram._compute_similarity([0.5, 0.5], [0.7, 0.3])
        expected = 0.9284766908852594
        self.assertEqual(similarity, expected)
if __name__ == '__main__':
    unittest.main()
