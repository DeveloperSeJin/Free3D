import openai
import nltk
from nltk import word_tokenize
from autocorrect import Speller
from TreatSentence import NER
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

class TextProcessing :

    def __init__(self, LLM_model, ner_model) :
    #사용시 키 입력
        #openai.api_key = key
        self.model = T5ForConditionalGeneration.from_pretrained(LLM_model, device_map="auto")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.ner = NER(ner_model)
        
        self.material_example = 'Wood, Metal, Plastic, Glass, Fabric, Leather'
        self.color_exmaple = 'Turquoise, Teal, Navy blue, Sky blue, Royal blue, Cobalt blue'
        self.size_exmaple = 'Tiny, Small, Petite, Long, Extensive, Far-reaching'
        self.design_exmaple = 'Modern, Contemporary, Minimalist, Scandinavian, Mid-century modern, Industrial'
        
    #사용자의 prompt를 토대로 category 및 detail을 추천해주는 함수
    
#     def request(self, prompt) :
#             response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You should help generate furniture text for stable diffusion."},
#                 {"role": "user", "content": prompt},
#             ]
#         )
    
#             return response.choices[0].message.content

    def request(self, prompt, num) :
        result = []
        
        for _ in range(num) :
            # prepare for the model
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
            # generate
            outputs = self.model.generate(input_ids, min_length=random.randrange(100, 130), max_length = random.randrange(150, 180), num_beams=random.randrange(1, 50), repetition_penalty=random.randrange(1, 4) / 10, temperature = random.randrange(8, 11) / 10, no_repeat_ngram_size = 1)
            result.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
    
        return result

    # object가 포함되어 있으면 object를 반환, 없으면 -1을 반환
#     def checkObject(self, prompt) :
        
#         spell = Speller(lang = 'en')
#         words = word_tokenize(prompt)
    
#         verifiedSentence = ' '.join(spell(word) for word in words)
#         print('verifiedSentence')
#         print(verifiedSentence)
        
#         result = self.ner.get_missing_tags(verifiedSentence)
        
#         return result, verifiedSentence


    #사용해야 하는 것 / input : 사용자 input, output : json
    def getAnswer(self, prompt) :
        if len(prompt.split()) == 1 :
            print('It\'s too short.')
            return 0
        
#         objs, verifiedSentence = self.checkObject(prompt)
        objs = self.ner.get_missing_tags(prompt)
        verifiedSentence = prompt
        
        if 'furniture' in objs:
            print('do not have furniture')
            return -1
        
        additional_detail = self.request('Supplementary explanation: ' + prompt, 1)
        print('additional_detail')
        print(additional_detail)
        
        recommend_dict = {'material': self.material_example, 'color': self.color_exmaple, 'size': self.size_exmaple, 'design': self.design_exmaple}
        
        recommend = ''.join('-It is recommended to represent ' + o + '\n example: ' + recommend_dict[o] + '\n\n' for o in objs)
        print('recommend')
        print(recommend)
        
        json_object = {
                    "recommend" :recommend
                }
        
        input_text = "Your role is a designer who explains furniture designs to users.\n\
The sentence that supplements and explains \"A white chair\" is as follows:\n\n\
- A white leather lounge chair with large wooden arms: This chair has a sleek and modern white leather upholstery that complements its wooden arms. The arms have a large size, ensuring comfortable and secure seating. The chair has a plush padded seat and backrest, ensuring optimal comfort and support.\n\n\
Enhance the following sentence, and provide a detailed explanation about this sentence\n\n\"" + verifiedSentence + "\"\n\n\
The format is as follows.\n\n\
- sentence : detail"
        
        while (True) :
            try :
                response = self.request(input_text, 3)
                print('response: ')
                if response[0].split(': ')[0] == response[1].split(': ')[0] or response[0].split(': ')[0] == response[2].split(': ')[0] or response[1].split(': ')[0] == response[2].split(': ')[0] :
                    raise Exception('duplication')
                for r in response :
                    if len(r.split(": ")) > 3:
                        raise Exception("format error")
                    print(r)
                    print('-'*50)
                detail_list = [r for r in response]
                #details = ''.join(dl + '\n\n' for dl in detail_list)
                
                
                index = 1
                detail_json = {}
                detail_json['detail0'] = {"prompt" : verifiedSentence, "detail" : additional_detail[0]}
                for d in detail_list :
                    detail_json['detail' + str(index)] = {"prompt" : d.split(': ')[0], "detail" : d.split(': ')[1]}
                    index += 1
                break
            except Exception as e:
                print(e)
                continue

  
        json_object['detail'] = detail_json
        json_string = json.dumps(json_object)
        return json_string