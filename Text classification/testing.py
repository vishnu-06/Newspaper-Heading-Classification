## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import model_selection,manifold
## for word embedding
import gensim
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
json_list = []
with open('News_Category_Dataset.json', mode='r', errors='ignore') as json_file:
    for element in json_file:
        json_list.append( json.loads(element) )
## print the first one
print(json_list[0])

df = pd.DataFrame(json_list)
## filter categories
df = df[ df["category"].isin(['ENTERTAINMENT','POLITICS','TECH']) ][["category","headline"]]
## rename columns
df = df.rename(columns={"category":"y", "headline":"X"})
## print 5 random rows
df.sample(5)
fig, ax = plt.subplots()
fig.suptitle("y", fontsize=12)
df["y"].reset_index().groupby("y").count().sort_values(by=
       "index").plot(kind="barh", legend=False,
        ax=ax).grid(axis='x')
plt.show()


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   characters and thenstrip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if stopwords is not None:
        lst_text = [word for word in lst_text if word not in stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

stopwords = nltk.corpus.stopwords.words("english")
df["X_clean"] = df["X"].apply(lambda x:
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
          stopwords=stopwords))
print(df.head())
x_train, x_test = model_selection.train_test_split(df, test_size=0.3)
## get target
y_train = x_train["y"].values
y_test = x_test["y"].values
corpus = x_train["X_clean"]

## create list of lists of unigrams
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1])
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300,
            window=8, min_count=1, sg=1, iter=30)
word = "data"
print(nlp[word])
fig = plt.figure()
# ## word embedding
total_words = [word] + [element[0] for element in nlp.wv.most_similar(word, topn=20)]
X = nlp[total_words]

## pca to reduce dimensionality from 300 to 3
pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
X = pca.fit_transform(X)
## create df
df_ = pd.DataFrame(X, index=total_words, columns=["x","y","z"])
df_["input"] = 0
df_["input"].iloc[0:1] = 1
## plot 3d
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_[df_["input"]==0]['x'],
           df_[df_["input"]==0]['y'],
           df_[df_["input"]==0]['z'], c="black")
ax.scatter(df_[df_["input"]==1]['x'],
           df_[df_["input"]==1]['y'],
           df_[df_["input"]==1]['z'], c="red")
ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
       yticklabels=[], zticklabels=[])
for label, row in df_[["x","y","z"]].iterrows():
    x, y, z = row
    ax.text(x, y, z, s=label)
plt.show()
## tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                     oov_token="NaN",
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
## create sequence
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
## padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq,
                    maxlen=15, padding="post", truncating="post")
sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
plt.show()
i = 0

## list of text: ["I like this", ...]
len_txt = len(x_train["X_clean"].iloc[i].split())
print("from: ", x_train["X_clean"].iloc[i], "| len:", len_txt)

## sequence of token ids: [[1, 2, 3], ...]
len_tokens = len(X_train[i])
print("to: ", X_train[i], "| len:", len(X_train[i]))

## vocabulary: {"I":1, "like":2, "this":3, ...}
print("check: ", x_train["X_clean"].iloc[i].split()[0],
      " -- idx in vocabulary -->",
      dic_vocabulary[x_train["X_clean"].iloc[i].split()[0]])

print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")
corpus = x_test["X_clean"]

## create list of uni-grams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0,
                                                             len(lst_words), 1)]
    lst_corpus.append(lst_grams)

## text to sequence with the fitted tokenizer
lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

## padding sequence
X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
                                            padding="post", truncating="post")
## start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word,idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] =  nlp[word]
    ## if word not in model then skip and the row stays all 0s
    except:
        pass

word = "data"
print("dic[word]:", dic_vocabulary[word], "|idx")
print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape,
      "|vector")

## input
x_in = layers.Input(shape=(15,))
## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)
y_out = layers.Dense(3, activation='softmax')(x)
## compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()
