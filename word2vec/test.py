import logging
import os
from collections import defaultdict

from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import tempfile

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))


def test_relation(directory, model):
    words = []
    with open('word2vec_test/' + directory) as file:
        for line in file:
            words.append(line.strip().split())
            if words[-1][0] not in model.wv.vocab or words[-1][1] not in model.wv.vocab:
                print('not in vocab ', words.pop())
    correct = 0
    apprx = 0
    counter = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            counter += 1
            pos = [words[i][0], words[j][1]]
            neg = [words[i][1]]
            ans = words[j][0]
            sims = model.wv.most_similar(positive=pos, negative=neg, topn=5)
            if ans == sims[0][0]:
                correct += 1
                print('yeees')
            if ans in [s for s, _ in sims]: apprx += 1

            print('real answer for pos=', pos, 'neg=', neg, 'ans=', ans, 'given answers: ', [s for s, _ in sims])
    print('correct answers= ', correct)
    print('apprx answers= ', apprx)
    print('all= ', counter)


def similar_words(model):
    with open('special_words') as f:
        for line in f:
            if line.strip() in model.wv.vocab:
                print(line)
                sim = model.wv.most_similar(positive=[line.strip()])
                print(line.strip(), '&', ', '.join([a for a, _ in sim]), '\\')


def cluster_vectors(model):
    word_vectors = model.wv.syn0norm
    # for i in range(50, 400, 25):
    num_clusters = 200
    # clustering = KMeans(n_clusters=num_clusters, n_init=20)
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    idx = clustering.fit_predict(word_vectors)
    print(clustering.n_components_)
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    d = defaultdict(list)
    for key, value in sorted(word_centroid_map.items()):
        d[value].append(key)
    with open('agglo', 'w') as f:
        for k, v in sorted(d.items()):
            f.write('----' + str(k) + '----')
            f.write('\n')
            f.write('، '.join(v))
            f.write('\n')


def main():
    w2v_model_saved = Word2Vec.load(os.path.join('word2vec_models', 'model.word2vec_300'))
    print(w2v_model_saved)
    # #
    # test_relation('relation_love', w2v_model_saved)
    # test_relation('relation_i', w2v_model_saved)
    # test_relation('relation_An', w2v_model_saved)
    # test_relation('relation_mi', w2v_model_saved)
    # test_relation('relation_past', w2v_model_saved)
    similar_words(w2v_model_saved)
    # print(w2v_model_saved.wv.most_similar(positive=['سرو','چشم'],negative=['قد']))
    # print(w2v_model_saved.wv.most_similar(positive=['سرو','ابرو'],negative=['قد']))
    print(w2v_model_saved.wv.most_similar(positive=['سرو', 'مو'], negative=['قد']))
    # print(w2v_model_saved.wv.most_similar(positive=['خورشید','قطره'],negative=['ذره']))

    cluster_vectors(w2v_model_saved)

    pass


if __name__ == '__main__':
    main()
