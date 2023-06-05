from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast, pipeline
import torch

class NER:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.recognizer = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        self.tag_dic = {'LABEL_0' : 'I-design',
         'LABEL_1' : 'B-design',
         'LABEL_2' : 'B-size',
         'LABEL_3' : 'I-material',
         'LABEL_4' : 'B-material',
         'LABEL_5' : 'B-color',
         'LABEL_6' : 'I-furniture',
         'LABEL_7' : 'O',
         'LABEL_8' : 'I-size',
         'LABEL_9' : 'I-color',
         'LABEL_10' : 'B-material'}
        
        self.unique_tag = ['B-color', 'I-color', 'B-furniture', 'I-furniture',                          'B-material', 'I-material', 'B-size', 'I-size', 'B-design', 'I-design', 'O']
    
    def return_dic(self, sentence) :
        result_tag = self.recognizer(sentence)
        
        for rt in result_tag:
            rt['entity'] = self.tag_dic[rt['entity']]
        
        return result_tag
    
    def get_tag(self, sentence):
        dic = self.return_dic(sentence)
        
        tag = set()
        for d in dic :
            #if d['score'] > 0.88:
            tag.add(d['entity'])
                
        return tag
    
    def get_missing_tags(self, sentence):
        tag = self.get_tag(sentence)
        recommend_tag = ['B-color', 'B-furniture', 'B-material', 'B-size', 'B-design']
        [recommend_tag.remove(t) for t in tag if t in recommend_tag]
        
        if len(recommend_tag) > 0 :
            result = [r.split('-')[1] for r in recommend_tag]
        else :
            result = []
        return result