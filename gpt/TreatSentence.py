from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast, pipeline
import torch

class NER:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.recognizer = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        self.tag_dic = {
         'LABEL_0' : 'B-Size',
         'LABEL_1' : 'B-Design',
         'LABEL_2' : 'I-Material',
         'LABEL_3' : 'I-Design',
         'LABEL_4' : 'B-Color',
         'LABEL_5' : 'I-Furniture',
         'LABEL_6' : 'I-Color',
         'LABEL_7' : 'I-Size',
         'LABEL_8' : 'B-Material',
         'LABEL_9' : 'B-Furniture',
         'LABEL_10' : '0',}
        
        self.unique_tag = ['B-Color', 'I-Color', 'B-Furniture', 'I-Furniture', 'B-Material', 'I-Material', 'B-Size', 'I-Size', 'B-Design', 'I-Design', 'O']
    
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
        recommend_tag = ['B-Color', 'B-Furniture', 'B-Material', 'B-Size', 'B-Design']
        [recommend_tag.remove(t) for t in tag if t in recommend_tag]
        
        if len(recommend_tag) > 0 :
            result = [r.split('-')[1] for r in recommend_tag]
        else :
            result = []
        return result