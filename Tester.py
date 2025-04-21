import unittest
from ex2_api import *
import tempfile


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.d = 400  # vector size
        self.context = 4  # 2 words after and before
        self.neg_samples = 2  # 2 negative samples for 1 positive sample
        self.step_size = 0.0001  # weights updating

        self.sentences = ['Mary enjoys cooking',
                          'She likes bananas',
                          'They speak English at work',
                          'The train does not leave at 12 AM',
                          'I have no money at the moment',
                          'Do they talk a lot',
                          'Does she drink coffee',
                          'You run to the party']

    def test_normalize_text(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp:
            temp.write("This is the first sentence. Here's the second sentence!")
            temp_path = temp.name

        try:
            # Call normalize_text on the temp file
            sentences = normalize_text(temp_path)

            # Check the output
            expected_sentences = ['this is the first sentence', 'heres the second sentence']
            self.assertEqual(expected_sentences, sentences)
        finally:
            # Clean up by deleting the temporary file
            os.remove(temp_path)

    def test_init(self):
        sentences = ['etay is a first sentence', 'etay is another sentence', 'a third sentence']
        sg = SkipGram(sentences, word_count_threshold=2)

        # Check word count dictionary
        expected_word_counts = {'etay': 2, 'sentence': 3}
        self.assertEqual(sg.word_count, expected_word_counts)

        # Check word:index mapping
        expected_word_index = {'etay': 0, 'sentence': 1}
        self.assertEqual(sg.word_index, expected_word_index)

    def test_cosine_similarity(self):
        cherry = [422, 8, 2]
        digital = [5, 1683, 1670]
        information = [5, 3982, 3325]

        # Check similarity between cherry and information
        similarity = SkipGram.cosine_similarity(cherry, information)
        self.assertAlmostEqual(similarity, 0.0185, places=2)

        # Check similarity between digital and information
        similarity = SkipGram.cosine_similarity(digital, information)
        print(similarity)
        self.assertAlmostEqual(similarity, 0.996, places=3)

    def test_preprocess_sentences(self):
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        le = sg.preprocess_sentences()

    def test_calculate_loss(self):
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        sg.vocab_size = 3
        y = np.array([0, 0, 1, 0])
        val = np.array([0.2, 0.7, 0.9, 0.3])
        expected_loss = 1.8891518152
        calculated_loss = sg.calculate_loss(y, val)
        print(calculated_loss)
        self.assertAlmostEqual(calculated_loss, expected_loss, places=5)
    #
    def test_learn_embeddings(self):
        # Set up test data and parameters
        step_size = 0.01
        epochs = 50
        early_stopping = 30
        model_path = "test_model.pkl"
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        T, C = sg.learn_embeddings(step_size, epochs, early_stopping, model_path)
        print(f"T shape:\n{T.shape}")
        print(f"C shape:\n{C.shape}")

    def test_combine_vectors(self):
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        T = np.array([[1, 2, 3], [4, 5, 6]])
        C = np.array([[7, 8, 9], [10, 11, 12]]).T
        model_path = "test_model.pkl"

        # Test combo = 0
        V = sg.combine_vectors(T, C, combo=0)
        np.testing.assert_array_equal(V, T)

        # Test combo = 1
        V = sg.combine_vectors(T, C, combo=1)
        np.testing.assert_array_equal(V, C.T)

        # Test combo = 2
        V = sg.combine_vectors(T, C, combo=2)
        np.testing.assert_array_equal(V, (T + C.T) / 2)

        # Test combo = 3
        V = sg.combine_vectors(T, C, combo=3)
        np.testing.assert_array_equal(V, T + C.T)

        # Test combo = 4
        V = sg.combine_vectors(T, C, combo=4)
        np.testing.assert_array_equal(V, np.concatenate((T, C.T), axis=1))

        # Test saving the model
        sg.combine_vectors(T, C, combo=0, model_path=model_path)
        self.assertTrue(os.path.exists(model_path))

        # Test invalid combo value
        with self.assertRaises(ValueError):
            sg.combine_vectors(T, C, combo=5)

    def test_get_closest_words(self):
        # Initialize a SkipGram object
        skip_gram = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        skip_gram.word_index = {'cat': 0, 'dog': 1, 'mouse': 2, 'parrot': 3, 'elephant': 4}
        skip_gram.index_word = {v: k for k, v in skip_gram.word_index.items()}
        skip_gram.vocab_size = len(skip_gram.word_index)

        # Assuming the embeddings (T matrix) is of shape (vocab_size, embed_dim)
        # We give similar vectors for 'cat', 'dog' and 'mouse' and distinct ones for 'parrot' and 'elephant'.
        skip_gram.T = np.array([[1, 0.9, 0.9, 0.1, 0.1],
                                [0.9, 1, 1, 0.2, 0.2],
                                [0.9, 0.85, 0.85, 0.15, 0.15]])

        # Now, test get_closest_words function
        closest_words = skip_gram.get_closest_words('cat', n=2)
        # 'cat' should be most similar to 'dog' and 'mouse'
        self.assertEqual(closest_words, ['dog', 'mouse'])

        # Test with a word not in the vocabulary
        closest_words = skip_gram.get_closest_words('tiger', n=2)
        self.assertEqual(closest_words, [])

    def test_find_analogy(self):
        # Initialize a SkipGram object
        skip_gram = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        skip_gram.word_index = {'king': 0, 'man': 1, 'woman': 2, 'queen': 3}
        skip_gram.index_word = {v: k for k, v in skip_gram.word_index.items()}
        skip_gram.vocab_size = len(skip_gram.word_index)

        # Assuming the embeddings (T matrix) is of shape (vocab_size, embed_dim)
        T = np.array([[0.8, 0.4, 0.2, 0.6],
                      [0.6, 0.3, 0, 0.3],
                      [0.4, 0.2, 0.6, 0.8]])
        skip_gram.T = T

        # Test the analogy "king - man + woman = queen"
        analogy_word = skip_gram.find_analogy('man', 'king', 'woman')
        self.assertEqual('queen', analogy_word)

        actual = skip_gram.test_analogy('man', 'king', 'woman', 'queen')
        return self.assertEqual(actual, True)
    # @staticmethod
    # def test_model():
    #     # Set up test data and parameters
    #     step_size = 0.01
    #     epochs = 50
    #     early_stopping = 30
    #     model_path = "harry_potter_model.pkl"
    #
    #     # Initialize your word2vec model
    #     # Load the text file and preprocess it
    #     sentences = normalize_text('corpora/harryPotter1.txt')
    #
    #     # Initialize a SkipGram object
    #     word2vec = SkipGram(sentences, neg_samples=2, word_count_threshold=3)
    #
    #     # Train the word2vec model on the text file
    #     T, C = word2vec.learn_embeddings(step_size, epochs, early_stopping, model_path)
    #
    #     print(f"T shape:\n{T.shape}")
    #     print(f"C shape:\n{C.shape}")
    #
    # def test_built_model(self):
    #     model = load_model('harry_potter_model.pkl')
    #     words = model.get_closest_words('cat', n=5)
    #     print(words)


if __name__ == '__main__':
    unittest.main()
