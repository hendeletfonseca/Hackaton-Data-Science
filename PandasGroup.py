import subprocess
import sys
import re
import pickle
import string
from urllib.request import urlopen

def install_dependencies():
    libs = ["scipy","pandas","scikit-learn","nltk","unidecode"]
    for lib in libs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    import nltk
    packages = ['stopwords', 'word_tokenize', 'RSLPStemmer','rslp','punkt']
    for package in packages:
        nltk.download(package)

def get_pickle_file(url):
    # https://stackoverflow.com/questions/53107052/can-we-load-pkl-files-from-an-external-url
    # https://medium.com/analytics-vidhya/save-and-load-your-scikit-learn-models-in-a-minute-21c91a961e9b
    return pickle.load(urlopen(url))

def remove_stopwords(text):
    from nltk.corpus import stopwords 
    # separating text in a list
    words = text.split(' ')
    
    # collecting stopword list
    stop_words = set(stopwords.words('portuguese')) 

    # filtering the text, removing the stopwords
    words_filtered = [w for w in words if w not in stop_words]
    
    # merging the word list into text again
    text_filtered = ' '.join(words_filtered)
    
    return text_filtered

#  function to apply stemming
def do_stem(text):
    from nltk.stem import RSLPStemmer
    # separating text into a word list
    words = text.split(' ')
    
    stemmer = RSLPStemmer()

    # stemming text
    words_stem = [stemmer.stem(w) for w in words if len(w)>0]
    
    # merging the word list into text again
    text_stem = ' '.join(words_stem)
    
    return text_stem

def tweet_processing(tweets):
    from nltk.tokenize import word_tokenize
    from unidecode import unidecode
    # removendo URLs
    tweets['text_preproc'] = tweets['text'].apply(lambda x: re.sub(r"http\S+", "", str(x)))
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: re.sub('@[^\s]+','',str(x)))
    # transformando em minúsculo
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: x.lower())
    # removendo acentuação
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: unidecode(x))
    # removendo caracteres de nova linha
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: x.replace('\r\n',' '))
    
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: remove_stopwords(x))
    
    # removendo pontuação dos tweets
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    # stemming
    tweets['text_preproc'] = tweets['text_preproc'].apply(lambda text: do_stem(text))
    
    # aplicando o tokenizer
    tweets['tokens'] = tweets['text_preproc'].apply(word_tokenize)
    return tweets

def main():
    install_dependencies()
    
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import CountVectorizer

    input_file = "teste.csv"
    out_file = "saida.csv"
    url = "https://gitlab.com/afego/HackathonCienciasDeDados/-/raw/main/svc_tfidf_sig.pkl"
    num_rep_url = "https://gitlab.com/afego/HackathonCienciasDeDados/-/raw/main/tfidf.pkl"
    num_train_url = "https://gitlab.com/afego/HackathonCienciasDeDados/-/raw/main/tfidf_train.pkl"

    classifier = get_pickle_file(url)
    num_repr = get_pickle_file(num_rep_url)
    num_train = get_pickle_file(num_train_url)
    num_repr.fit(num_train)

    df = pd.read_csv(input_file, delimiter=';')
    df = tweet_processing(df)

    num_repr_predict = num_repr.transform(df['text_preproc']).toarray()
    # y_pred = classifier.predict(num_repr_predict)
    prob = classifier.predict_proba(num_repr_predict)
    df_prob = pd.DataFrame(prob,columns=['negativo','neutro','positivo'])

    df_prob[['negativo','neutro','positivo']].to_csv(out_file, sep=';', index=False)
    return

if __name__ == '__main__':
    main()