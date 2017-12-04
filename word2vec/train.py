import logging
import os

from gensim.models import Word2Vec
from hazm import Normalizer, word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class PoemSentences(object):
    def __init__(self, poems_path):
        self.poems_path = poems_path
        self.normalizer = Normalizer()

    def __iter__(self):
        for poem_file in os.listdir(self.poems_path):
            for sentence in open(os.path.join(self.poems_path, poem_file)):
                yield word_tokenize(self.normalizer.normalize(sentence.replace('هٔ', 'ه')))


def train_model(poems_path):
    sentences = PoemSentences(poems_path)
    w2v_model = Word2Vec(sentences=sentences, size=250, window=5, min_count=10, workers=5, sg=1,
                         iter=100, negative=5, hs=0, max_vocab_size=10000)
    w2v_model.save(os.path.join('word2vec_models', 'model.word2vec_250_ng'))


if __name__ == '__main__':
    train_model('poems')
