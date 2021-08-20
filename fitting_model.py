#!/usr/bin/env python3

#Logistic regression model to classify if an answer is correct, given a question. Model is trained on a dataset of various answers (correct and wrong).

#useful imports : 
import pandas as pd 
import numpy as np 
import texthero as hero
from nltk.corpus import stopwords 
from matplotlib import pyplot as plt 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


#constant definition:
datafile = pd.read_csv("input/trainerDataset.csv")
stopWords = set(stopwords.words('french'))


############################
#Script starts here#########
############################



def classifier():
    #Selecting 2nd column and affecting it
    answers = datafile.iloc[:,1] 
    #Cleaning the strings (removing uppercase, punctuation...)
    answers = string_cleanser(answers)
    print(answers)
    #vectorizing the strings in the pandas list:
    answersVec = string_to_vec(answers)
    dataVec = np.array(answersVec).tolist()
    np.savetxt('output/trained_model.csv',dataVec)


def string_to_vec(dataset):
    wordDataset = [TaggedDocument(doc.split(' '), [i]) for i, doc in enumerate(dataset)]
    model = Doc2Vec(vector_size=64,window=4,min_count=1,workers=8,epochs=60)
    model.build_vocab(wordDataset)
    model.train(wordDataset, total_examples=model.corpus_count, epochs=model.epochs)
    vecDataset = [model.infer_vector((dataset[i].split(' '))) for i in range(0,len(dataset))]
    return vecDataset 

def string_cleanser(dataset):
    dataset = hero.lowercase(dataset)
    dataset = hero.remove_stopwords(dataset,stopWords)
    dataset = hero.remove_digits(dataset)
    dataset = hero.remove_punctuation(dataset)
    dataset = hero.remove_diacritics(dataset)
    dataset = dataset.str.replace(' +',' ')
    dataset = dataset.str.lstrip()
    return dataset

classifier()