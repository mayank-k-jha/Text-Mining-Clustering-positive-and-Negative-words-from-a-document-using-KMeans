from textblob import TextBlob
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn import cluster



# Let's create our feature vector as Kmeans works with numerical values only
# thus we need to convert our text data into numeric features in some way which should 
# prove usful to our learning algorithm and it can easily find pattern in it.
# So, i will be using polarity score of the extracted adjectives as our feature

def create_polarity_vector(all_adjectives):
    # NLTK Vader sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    #Taking compound score as one feature, (compund score = -ve indicates negative sentiments)
    #(compund score = +ve indicates positive sentiments, ie compound value is normalization b/w +ve and -ve)
    feature_vector=[1 if sid.polarity_scores(i)['compound']>=0 else -1 for i in all_adjectives]
    
    return feature_vector


# utility function to read text file
def polarity_sets_file(input_file):
    with open(input_file, 'r') as f:
        input_text = f.read()
        return input_text

# utility function to extract all adjectives from text document
def extract_adjectives(text):
    tokenized_text= nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(tokenized_text)
    all_adjectives = list({i[0] for i in tagged_text if i[1] == 'JJ'})
    return all_adjectives

# Cleaning up text 
def clean_text(text):    
    
    #stopword removal
    stop_word=list(stopwords.words('english'))
    toke=list(text.split(' '))
    text = ' '.join([i for i in toke if i not in stop_word])
    
    #punctutaion removal
    t_lator=str.maketrans('','',string.punctuation)
    text=text.translate(t_lator)
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)    
    
    #removing special symbol
    for i in '“”—':
        text = text.replace(i, ' ')
        
    return text



# Reading our file
text = polarity_sets_file('test.txt')

# cleaning our text data
text = clean_text(text)

# Getting adjectieves from our document
adjectives = extract_adjectives(text)

# Preparing our feature vector
features = create_polarity_vector(adjectives)

# Making clusters using KMeans
feature_ = np.array(features).reshape(-1,1)
kmeans = cluster.KMeans(n_clusters=2, n_init=200)
kmeans.fit(feature_)
labels = kmeans.predict(feature_)

# Identifying positive and negative terms

type1_terms,type2_terms = [], []

for i,j in zip(adjectives,labels):
    if j == 0 :
        type1_terms.append(i)
    else:
        type2_terms.append(i)
        
print(type1_terms)

print(type2_terms)

