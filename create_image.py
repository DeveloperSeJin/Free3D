# Todo: 전체 코드 정리
# Todo: text module, image module 별도 파일로 구분하기
from gpt.Getprompt import TextProcessing_gpt, TextProcessing_T5
from tap import Tap
import time
import datetime as dt
from typing import Optional
from diffusion import Diffuse
import os
import sys
import subprocess
from pathlib import Path

try:
    from rembg import remove
except ImportError:
    print('Please install rembg with "pip install rembg"')
    sys.exit()

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Text Processing 모델
class CustomLanguageModel():
    def __init__(self, opt):
        self.model = None
        self.model_name = opt.model_name
        
        if self.model_name == "T5":
            self.model = TextProcessing_T5(
                ner_model="DeveloperSejin/NER_for_furniture_3D_object_create",\
                model_name="DeveloperSejin/Fine_Tuned_Flan-T5-large_For_Describe_Furniture"
            )
        elif self.model_name == "GPT":
            self.model = TextProcessing_gpt(
                ner_model="DeveloperSejin/NER_for_furniture_3D_object_create",\
                key=opt.key
            )
        else:
            raise NotImplementedError(f'--model_name {opt.model_name} is not implemented!')
    
    def get_prompt(self, user_prompt, opt):
        model_name = self.model_name
        model = self.model
        answer = None
        start = time.time()
        
        if model_name == 'GPT':
            print("GPT", end=' ')
            answer = model.getAnswer(prompt = user_prompt)
            del model
            
        else:
            print("T5", end=' ')
            
            answer = model.getAnswer(prompt = user_prompt)
            del model
        
        if(answer == -1):
            answer = {"recommend": "가구가 아닌 데이터는 금전적인 문제로 Fine tuning이 되지 않아 문장 생성을 도와드릴 수 없습니다. 기본 문장으로 생성을 진행해도 괜찮습니까?","detail": {"detail0": {"prompt": user_prompt, "detail": "Original Prompt"}}}
         
        return answer

def generate_image(opt, prompt):
    org_image = None # Orginal image before background removal
    
    if opt.model_name == 'GPT':
        org_image = Diffuse.run(prompt)
    elif opt.model_name == "T5":
        org_image = Diffuse.run2(prompt)
        
    # if image == None:
    #     raise ValueError("Image is not generated!")        
    
    rb_image = remove(org_image, alpha_matting=False)
        
    return org_image, rb_image

def save_image(opt, org_image, rb_image, index):
    # ID = opt.model_name
    path = None
    
    if opt.save_path is not None:
        path = opt.save_path
    else:
        path = opt.workspace + f'/result_{opt.category}'
        if not os.path.exists(path):
            os.makedirs(path)
        
    if opt.save_org_img:
        org_image.save(f'{path}/original_img_{opt.category}_{index+1}.png', 'png')
    rb_image.save(f'{path}/generated_{opt.category}_{index+1}.png', 'png')
    
def save_prompt(opt, prompt):
    path = None
    if opt.save_path is not None:
        path = opt.save_path
    else:
        path = opt.workspace + f'/result_{opt.category}'
        if not os.path.exists(path):
            os.makedirs(path)
        
    with open(f'{path}/prompt_{opt.category}.txt', 'w') as f:
        f.write(''.join(prompt))
        
def main():
    # TO dO
    # 3. 셸 스크립트로 테스트 자동화?
    opt = Options().parse_args()
    print(opt)
    (Path(opt.workspace) / 'command.txt').write_text(subprocess.list2cmdline(sys.argv[1:]))
    text_model = CustomLanguageModel(opt)
    prompts = []
    
    # CLI에서 입력받은 이미지 개수만큼 이미지 만들기
    for i in range(opt.img_num):
        print(f"processing {i}th prompt and img")
        
        # 프롬프트 엔지니어링(LLM을 통한 프롬프트 자동 생성)
        json_string, response = text_model.get_prompt(opt.text, opt)
        
        # 완성된 프롬프트 중 랜덤으로 하나 가져오기
        prompt = response[int(time.time())%3]
        if opt.split_prompt:
            prompt = prompt.split(":")[0]
        
        # 완성된 프롬프트로 이미지 생성하기
        org_img, rb_img = generate_image(opt, prompt)
        
        # 프롬프트 저장을 위해 리스트에 prompt 저장
        prompts.append(f"{i+1}. {prompt}\n")
        
        # 이미지 저장하기
        if opt.save_img:
            save_image(opt, org_img, rb_img, i)
        
        print(f"Processing of {i}th prompt and img is successfully finish") 
            
    # 프롬프트 저장하기        
    if opt.save_prompt:
        save_prompt(opt, prompts)
    
class Options(Tap):
    """
    Todo: 
      1. args 설명 주석 추가
      2. Options.py 파일로 메인 코드에서 분리하기
    """
    text: str = None
    key: str = None
    model_name: str
    model_list: list = ["T5", "t5", "gpt", "GPT"]
    workspace: str = None
    img_num: int = None
    split_prompt: bool = True
    
    # saving
    save_img: bool = True     # save backrgound removal img
    save_path: str = None
    save_prompt: bool = True
    save_org_img: bool = True # save original image
    category: str = None
                   
    def process_args(self):
        # prompt 추천, 이미지 생성에 쓰일 prompt 미입력 시 error 발생
        if self.text is None:
            raise ValueError('text args is required, but empty')
            
        if self.category is None:
            raise ValueError('category args is required, but empty')
            
        # 모델 이름이 GPT일 경우 openai의 api 키를 입력받아야 함. 키 미입력 시 error 발생
        if self.model_name.startswith('GPT'):
            if self.key is None:
                raise ValueError('API key is required, but empty')
        
        # 입력받은 model_name이 지원하지 않는 model일 경우 error 발생
        if self.model_name not in self.model_list:
            raise ValueError(f'{self.model_name} is incorrect model_name. See this available list {self.model_list}')
        
        # model name을 대문자로 변경 ex) t5 -> T5, gpt -> GPT
        self.model_name = self.model_name.upper()
        
        # 생성할 이미지 개수를 입력하지 않으면 default로 1개 생성
        if self.img_num is None:
            self.img_num = 1
        
        # working directory 미입력 시 새로 working dir 생성
        if self.workspace is None:
            name_from_datetime = dt.datetime.now().strftime("%m-%d-%H-%M")
            workspace = f'./images/{name_from_datetime}_workspace'
            if not os.path.exists(workspace):
                os.makedirs(workspace)
                
            self.workspace = workspace

if __name__ == "__main__":
    main()