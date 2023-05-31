import gensim
import pandas as pd

df = pd.read_json("Dataset.json",lines=True)
print(df.isna().sum())

review_text = df['reviewText'].apply(gensim.utils.simple_preprocess)

print(review_text)

model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)

model.build_vocab(review_text, progress_per=1000)

model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

model.wv.most_similar("bad")
