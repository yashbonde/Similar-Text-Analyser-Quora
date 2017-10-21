# Similar-Text-Analyser-Quora
A Simple Neural Network based Text Similarity Analyser.

Our input data will be Quora (competition)[https://www.kaggle.com/c/quora-question-pairs] hosted on Kaggle, you can get the data from there after accepting the rules for competition.

We are using word2vec in skip grams to embed our inputs and then feeding them through an Neural Network Ensemble comprising of multi layer (differnt models and their results will be put here) LSTM-RNN for converting the sentences into final vectors and then passing those vectors through a deep fully connected layer to predict whether they were similar or not.
