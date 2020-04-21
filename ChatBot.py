# Import Libraries
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

# Ignore Any Warning messages
warnings.filterwarnings("ignore")

# Download Packages from nltk
nltk.download("punkt" , quiet=True)
nltk.download("wordnet", quiet=True)

#Get Article URL
article = Article("https://www.nerdwallet.com/blog/investing/stock-trading-how-to-begin/")
article.download()
article.parse()
article.nlp()
corpus = article.text

#print(corpus)

# Tokenization
text = corpus
sent_token = nltk.sent_tokenize(text) #Converting the text to list of sentences

#Print list of sentences
#print(sent_token)

#Create a Dictionary to remove Punctuation
remove_punct_dic = dict( (ord(punct),None) for punct in string.punctuation)

#Print the punctuation
#print(remove_punct_dic)

#Function to remove Punctuation
def LemNormalize(text):
    return nltk.word_tokenize(text)

#Print the Tokenizaion Text
print(LemNormalize(text))