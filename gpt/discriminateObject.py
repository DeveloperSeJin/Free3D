import nltk
from nltk import word_tokenize
from autocorrect import Speller
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

#object 종류가 늘어남에 따라 변경해야 함
target = 'chair'

# object가 포함되어 있으면 object를 반환, 없으면 -1을 반환
def discriminateObject(prompt) :
  spell = Speller(lang = 'en')
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(prompt)
  for word in words :
    corrected_word = lemmatizer.lemmatize(spell(word), 'v')
    if target == corrected_word :
      return corrected_word
  return '-1'