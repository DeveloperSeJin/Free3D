from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast, pipeline
import torch

class NER:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.recognizer = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        self.tag_dic = {'LABEL_0' : 'B-design',
         'LABEL_1' : 'O',
         'LABEL_2' : 'B-material',
         'LABEL_3' : 'I-furniture',
         'LABEL_4' : 'B-color',
         'LABEL_5' : 'B-materialO',
         'LABEL_6' : 'I-size',
         'LABEL_7' : 'B-size',
         'LABEL_8' : 'I-material',
         'LABEL_9' : 'I-color',
         'LABEL_10' : 'B-furniture',
         'LABEL_11' : 'I-design'}
        
        self.unique_tag = ['B-color', 'I-color', 'B-furniture', 'I-furniture',                          'B-material', 'I-material', 'B-size', 'I-size', 'B-design', 'I-design', 'O']
    
    def return_dic(self, sentence) :
        result_tag = self.recognizer(sentence)
        
        for rt in result_tag:
            rt['entity'] = self.tag_dic[rt['entity']]
        
        return result_tag
    
    def get_tag(self, sentence):
        dic = self.return_dic(sentence)
        
        return set([d['entity'] for d in dic])
    
    def get_missing_tags(self, sentence):
        tag = self.get_tag(sentence)
        recommend_tag = ['B-color', 'B-furniture', 'B-material', 'B-size', 'B-design']
        [recommend_tag.remove(t) for t in tag if t in recommend_tag]
        
        if len(recommend_tag) > 0 :
            result = [r.split('-')[1] for r in recommend_tag]
        else :
            result = []
        return result