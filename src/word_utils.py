import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json


def load_dictionary():
    filename = '../data/raw/dict/cmudict-0.7b'
    skip_chr = '[^A-Z-]'
    word_dct = []

    with open(filename, encoding='ISO-8859-1') as f:
        for line in f:
            if line[0:3] == ';;;': # Remove comments from file
                continue

            word, phne = line.strip().split('  ') # Split the line into word and phonemes

            if re.search(skip_chr, word): # Ensure only letters and hyphens are included
                continue
            else:
                data = (word, phne)
                word_dct.append(data)

    # Turn our list into a Pandas DataFrame
    df = pd.DataFrame(word_dct, columns=['word','phonetics'])

    print('Size of word dictionary:',len(df))
    return df


def single_word_conversion(df, term, encd):
    try:
        term = term.upper()
        data = df[df['word'] == term]
        
        for index, row in data.iterrows():
            word = list(row['word'])
            targ = row['phonetics'].split(' ')
            print('Source:', word)
            print('Target:', targ)
            print('#### TOKENISING ####')
            
            print('Source:', encd.transform(word))
            print('Target:', encd.transform(targ))
    except:
        print('Failed')
        
        
# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        vector = [0 for _ in range(max_int)]
        vector[seq] = 1
        Xenc.append(vector)
    yenc = list()
    for seq in y:
        vector = [0 for _ in range(max_int)]
        vector[seq] = 1
        yenc.append(vector)
    return Xenc, yenc

def generate_sample(df, inp, oup, lbl_enc, n_chr):
    t = df.sample(n=20000)

    X, y = [], []

    for index, row in t.iterrows():
        a = list(row['word'])
        b = ['<PAD>'] * (inp - len(a))
        a = np.hstack([a,b])
        a = lbl_enc.transform(a)

        c = row['phonetics'].split(' ')
        d = ['<PAD>'] * (oup - len(c))
        c = np.hstack([c,d])
        c = lbl_enc.transform(c)

        a, c = one_hot_encode(a, c, n_chr)
        
        X.append(a)
        y.append(c)

    X = np.array(X)
    y = np.array(y)

    return X, y


def new_word(input_word, inp, lbl_enc, n_chr):
    word = list(input_word.upper())
    b = ['<PAD>'] * (inp - len(word))
    word = np.hstack([word,b])
    word = lbl_enc.transform(word)
    
    Xenc, X = list(), []
    for seq in word:
        vector = [0 for _ in range(n_chr)]
        vector[seq] = 1
        Xenc.append(vector)
    X.append(Xenc)
    X = np.array(X)
    return X


def load_and_predict(input_df, field):
    # load json and create model
    json_file = open('../model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../model/model.h5")
    print("Loaded model from disk")

    lbl_enc = LabelEncoder()
    lbl_enc.classes_ = np.load('../model/encoder.npy')

    master = []
    for index, row in input_df.iterrows():
        input_word = row[field]
        X = new_word(input_word, 34, lbl_enc, 81)

        result = loaded_model.predict(X, batch_size=1, verbose=0)

        output = []
        for i in result:
            for a in i:
                a = a.tolist() 
                ind = a.index(max(a))
                output.append(ind)

        word = lbl_enc.inverse_transform(output)
        word = [y for y in word if y != '<PAD>']
        data = " ".join(word)
        master.append(data)

    input_df['phonetic_predict'] = master
    return input_df