from collections import defaultdict
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

df = pd.read_csv('post_top_comment.csv')

# cleaning data
df = df.dropna().reset_index(drop=True)
df.isnull().sum()

sentences = [str(doc).split() for doc in df['post_text']]

for s in sentences:
    for ch in s:
        if not ch.isalpha():
            s.remove(ch)

phrases = Phrases(sentences, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sent = bigram[sentences]

# word frequency
word_freq = defaultdict(int)
for s in sent:
    for i in s:
        word_freq[i] += 1

sorted(word_freq, key=word_freq.get, reverse=True)[:30]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
model.init_sims(replace=True)

# finding words similar
model.wv.most_similar(positive=['family'])
