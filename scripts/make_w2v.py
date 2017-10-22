import pandas as pd # dataframe
import numpy as np # matrix maths
from tqdm import tqdm # progress bar
from helper import text_to_wordlist # NLP
import gensim.models.word2vec as word2vec # word2vec model
import multiprocessing # cpu_count
import os # saving the model

# loading the questins
train = pd.read_csv('~/Kaggle_Quora/train.csv')
test = pd.read_csv('~/Kaggle_Quora/test.csv')

# checking for total number of nan values
print('[*]Total nan values(train):', train.isnull().sum())
print()
print('[*]Total nan values(test):', train.isnull().sum())
# so there are 2 nan values in question2, we will simply remove them
# as we have sufficient data
train = train.dropna()
test = test.dropna()
print()
print('[*]train.shape:', train.shape)
print('[*]test.shape:', test.shape)

q1_train = train['question1'].values.tolist()
q2_train = train['question2'].values.tolist()
q1_test = test['question1'].values.tolist()
q2_test = test['question2'].values.tolist()

# for training data
q1_train_sent = []
q2_train_sent = []
for i in tqdm(range(len(q1_train))):
    q1_train_sent.append(text_to_wordlist(q1_train[i]))
    q2_train_sent.append(text_to_wordlist(q2_train[i]))
    
# for testing data
q1_test_sent = []
q2_test_sent = []
for i in tqdm(range(len(q1_test))):
    q1_test_sent.append(text_to_wordlist(q1_test[i]))
    q2_test_sent.append(text_to_wordlist(q2_test[i]))

q_total = q1_train_sent
q_total += q2_train_sent
q_total += q1_test_sent
q_total += q2_test_sent

# word2vec parameters
e_dim = 300 # embedding dimension
min_word_count = 1 # minimum number of times a word comes so that it is registered
num_workers = multiprocessing.cpu_count() # total number of workers
context_size = 10 # number of words before and after the focus word
# context_size is also the maximum distance between the current and predicted word within a sentence.
downsampling = 1e-5 # threshold for configuring which higher-frequency words are randomly downsampled
seed = 1 # seed value, helps with remaking the model
sg = 0 # if sg = 0 CBOW is used, if sg = 1 skip-grams is used, default 0
epochs = 5 # number of iters or epochs over the corpus

# Making the word2vec model --> skip_grams
w2vector_sg = word2vec.Word2Vec(sg = 1, seed = seed, size = e_dim, workers = num_workers,
                             min_count = min_word_count, window = context_size,
                             sample = downsampling, iter = epochs)
# building the vocabulary
w2vector_sg.build_vocab(q_total)
print("Word2Vec vocabulary length:", len(w2vector_sg.wv.vocab))
# training the model
w2vector_sg.train(q_total, total_examples = w2vector_sg.corpus_count, epochs = w2vector_sg.iter)

# saving the model
if not os._exists('trained'):
    os.makedirs('trained')
w2vector_sg.save(os.path.join('trained', 'w2vector_sg.w2v'))
