import openai
import nltk
from nltk import word_tokenize
from autocorrect import Speller
from nltk.stem.wordnet import WordNetLemmatizer

class gpt :

    def __init__(self, key) :
    #사용시 키 입력
        openai.api_key = key

    #사용자의 prompt를 토대로 category 및 detail을 추천해주는 함수
    def request(self, text) :
        response = openai.Completion.create(
            model = "davinci:ft-personal-2023-05-01-20-32-10",
            max_tokens = 300,
            prompt = text
        )
    
        return response.choices[0].text

    # object가 포함되어 있으면 object를 반환, 없으면 -1을 반환
    def checkObject(self, prompt) :
        nltk.download('wordnet')
        #object 종류가 늘어남에 따라 변경해야 함
        target = 'chair'
    
        spell = Speller(lang = 'en')
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(prompt)
    
        for word in words :
            corrected_word = lemmatizer.lemmatize(spell(word), 'v')
            if target == corrected_word :
                return corrected_word
        return '-1'


    import json

    #사용해야 하는 것 / input : 사용자 input, output : json
    def getAnswer(self, prompt) :
        obj = self.checkObject(prompt)
        
        if obj == '-1' :
            return -1
    
        text = 'I want to get a more expressed example ' + prompt + '. Recommend a some example with detailed description? Please answer in the format below\n\
        1.summary : detail\n\
        2.summary : detail\n\
        3.summary : detail\n\
        4.summary : detail\n\
        5.summary : detail'
    
        recommend = '1. Color\n2.Size\n3. Materia\n4. Design'
        detail = self.request(text)
    
        print(detail)
        details = detail.split('\n\n')

        json_object = {
            "recommend" :recommend
        }

        index = 0
        detail_json = {}

        for d in details :
            detail_json['detail' + str(index)] = {"prompt" : d.split(': ')[0], "detail" : d.split(': ')[1]}
            index += 1
  
        json_object['detail'] = detail_json
        json_string = json.dumps(json_object)
        return json_string