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
from sklearn import model_selection,manifold, preprocessing, metrics
## for word embedding
import gensim
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing, backend as K
from imblearn.over_sampling import RandomOverSampler
import time

json_list = []
with open('News_Category_Dataset.json', mode='r', errors='ignore') as json_file:
    for element in json_file:
        json_list.append( json.loads(element) )
## print the first one
print("\n\n\n")
print("First Record in Json Format : ",json_list[0],"\n")

df = pd.DataFrame(json_list)
## filter categories
df = df[ df["category"].isin(['POLITICS','WELLNESS','ENTERTAINMENT','TRAVEL','STYLE & BEAUTY','QUEER VOICES','FOOD & DRINK','SPORTS','HOME & LIVING', 'WEDDINGS']) ][["category","headline"]]
print()
print("Category Counts:")
print("POLITICS :"+str(df["category"].value_counts()['POLITICS']))
print("WELLNESS :"+str(df["category"].value_counts()['WELLNESS']))
print("ENTERTAINMENT :"+str(df["category"].value_counts()['ENTERTAINMENT']))
print("TRAVEL :"+str(df["category"].value_counts()['TRAVEL']))
print("STYLE & BEAUTY :"+str(df["category"].value_counts()['STYLE & BEAUTY']))
print("QUEER VOICES :"+str(df["category"].value_counts()['QUEER VOICES']))
print("FOOD & DRINK :"+str(df["category"].value_counts()['FOOD & DRINK']))
print("SPORTS :"+str(df["category"].value_counts()['SPORTS']))
print("HOME & LIVING :"+str(df["category"].value_counts()['HOME & LIVING']))
print("WEDDINGS :"+str(df["category"].value_counts()['WEDDINGS']))
print()

print("Total Records :"+str(df.shape[0]))
print()
print(df.head())
## rename columns
df = df.rename(columns={"category":"y", "headline":"X"})
## print 5 random rows
print("\n 5 rows of Dataframe:")
print(df.sample(5))
fig, ax = plt.subplots()
fig.suptitle("Input Data", fontsize=12)
df["y"].reset_index().groupby("y").count().sort_values(by=
       "index").plot(kind="barh", legend=False,
        ax=ax).grid(axis='x')
plt.xlabel('Number of Data Items')
plt.show()


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, stopwords=None):

    ## clean (convert to lowercase and remove punctuations and   characters and then strip)
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

## code attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

## input
x_in = layers.Input(shape=(15,))
## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)
## apply attention
x = attention_layer(x, neurons=15)
## 2 layers of bidirectional GRU
x = layers.Bidirectional(layers.GRU(units=15, dropout=0.2,
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(units=15, dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(10, activation='softmax')(x)#will change with number of categories
## compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()
## encode y
dic_y_mapping = {n:label for n,label in
                 enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])
## training model
tic = time.perf_counter()
training = model.fit(x=X_train, y=y_train, batch_size=256,
                     epochs=50, shuffle=True, verbose=2,
                     validation_split=0.3)

toc = time.perf_counter()
print(f"Training time : {toc - tic:0.4f} seconds")

## plot loss and accuracy
metricss = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metricss:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metricss:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()
predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in
             predicted_prob]

## select observation
i = 0
txt_instance = x_test["X"].iloc[i]
## check true value and predicted value
print("Input Data: ",txt_instance,"Actual Category:", y_test[i])
print( "Predicted :", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))


def evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5)):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()
evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15,5))