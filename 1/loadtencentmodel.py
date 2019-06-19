from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = KeyedVectors.load_word2vec_format('tencentlab.txt', binary=False)
model.save('tencent.model')