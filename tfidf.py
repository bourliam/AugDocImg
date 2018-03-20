import operator
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:
  
  def __init__(self, stopwords):
    self.stopwords = stopwords
  
  def remove_stop_words(self, doc):
    filtered_words = [word for word in doc.split() if word not in self.stopwords]
    text = ""
    for word in filtered_words:
        text = text + word + ' '
    return text


  def processFiles(self, filenames):
    documents = []
    for file in filenames:
        f = open(file,'r').read().lower()
        doc = self.remove_stop_words(f)
        documents.append(doc)
    return documents

  def keywords(self, text, numb_to_retain = 5):
    docnames = [file for file in glob.glob("tfidfData/*.txt")]
    docnames += [file for file in glob.glob("texts/*")]
    print('\nFilenames of the document collection: ')
    print(docnames)

    documents = []
    documents.append(self.remove_stop_words(text.lower()))
    documents += self.processFiles(docnames)

    tfidf_vectorizer = TfidfVectorizer(stop_words=self.stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    words = tfidf_vectorizer.get_feature_names()
    vectors = tfidf_matrix.todense().tolist()

    # We are interested only in the first doc
    vector = vectors[0]

    # Bow feature vector as list of tuples
    words_weights = zip(words,vector)
    # keep only non zero values (the words in the document)
    nonzero = [my_tuple for my_tuple in words_weights if my_tuple[1]!=0]
    # rank by decreasing weights
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    # retain top 'my_percentage' words as keywords
    

    keywords = [my_tuple for my_tuple in nonzero[:numb_to_retain]]
  
    return keywords
    


