import openai
import nltk
from nltk import word_tokenize
from autocorrect import Speller
from nltk.stem.wordnet import WordNetLemmatizer
from TreatSentence import NER
import json

nltk.download('wordnet')

class gpt :

    def __init__(self, key, ner_model) :
    #사용시 키 입력
        openai.api_key = key
        
        self.ner = NER(ner_model)
        
    #사용자의 prompt를 토대로 category 및 detail을 추천해주는 함수
    def request(self, prompt) :
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You should help generate text for stable diffusion."},
                {"role": "user", "content": prompt},
            ]
        )
    
            return response.choices[0].message.content

    # object가 포함되어 있으면 object를 반환, 없으면 -1을 반환
    def checkObject(self, prompt) :
        #object 종류가 늘어남에 따라 변경해야 함
        none_list = self.ner.get_missing_tags(prompt)
    
        spell = Speller(lang = 'en')
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(prompt)
    
        verifiedSentence = ' '.join(lemmatizer.lemmatize(spell(word), 'v') for word in words)
        
        result = self.ner.get_missing_tags(verifiedSentence)
        return result, verifiedSentence


    #사용해야 하는 것 / input : 사용자 input, output : json
    def getAnswer(self, prompt) :
        objs, verifiedSentence = self.checkObject(prompt)
        
        if 'furniture' in objs:
            print('do not have furniture')
            return -1
    
        text = 'I want to get a more expressed example ' + verifiedSentence + '. Recommend a some example with detailed description? Please answer in the format below\n\
        1.summary : detail\n\
        2.summary : detail\n\
        3.summary : detail\n\
        4.summary : detail\n\
        5.summary : detail'
        
        recommend = ''.join('-' + o + '\n\n' for o in objs)
        
        while (True) :
            response = self.request(text)
            try :
                detail = response.split('\n\n')
                detail_list = [d for d in detail if d[0].isdigit()]
                details = ''.join(dl + '\n\n' for dl in detail_list)
                json_object = {
                    "recommend" :recommend
                }
                
                index = 0
                detail_json = {}
                for d in details.split('\n\n')[:-1] :
                    detail_json['detail' + str(index)] = {"prompt" : d.split(': ')[0], "detail" : d.split(': ')[1]}
                    index += 1
                break
            except Exception as e:
                print(e)
                continue

  
        json_object['detail'] = detail_json
        json_string = json.dumps(json_object)
        return json_string