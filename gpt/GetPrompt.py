import openai

openai.api_key = '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'

def getPrompt(discription) :
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Stable Diffusion으로 " + discription + "에 관한 사진을 뽑고 싶은데 적절한 태그나 prompt를 영어로 줄 수 있을까?    아래와 같은 형식으로 부탁해xxx,xxx,xxx,xxxxx,xxxxxx,xxx,xxx,..."},
      ]
  )
  return response.choices[0].message.content
