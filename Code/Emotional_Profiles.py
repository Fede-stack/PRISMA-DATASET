
!pip install datasets
!pip install bitsandbytes
!pip install torch
!pip install transformers
!pip install accelerate
!pip install einops
!pip install peft
#!pip install trl
#!pip install ipywidgets==7.7.1

import os

def get_json_paths(directory_path):
    json_paths = []


    if not os.path.isdir(directory_path):
        print(f"La directory {directory_path} non esiste.")
        return json_paths


    for filename in os.listdir(directory_path):

        full_path = os.path.join(directory_path, filename)


        if os.path.isfile(full_path) and filename.lower().endswith('.json'):
            json_paths.append(full_path)

    return json_paths

# json_files = get_json_paths('')
# emotional_files = get_json_paths('')



import re
import torch
import transformers
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')
from transformers.utils import logging
logging.set_verbosity(transformers.logging.CRITICAL)

import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import login
login()

model_name = 'DeGra/RACLETTE-v0.2'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
  )


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #revision=rev,
    quantization_config=bnb_config,
    #use_flash_attention_2=True,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

generation_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

#max_limited_chars:  number of selected characters that will appear in the string
#stop at max: False includes rest of string until next
def filter_limit_chars(answer,limit_chars,max_limited_chars=2,stop_at_max=False,verbose=False):
      count=0
      index=[0]

      for separator in limit_chars:
          separator_count=answer.count(separator)
          count+=separator_count
          i=0
          for _ in range(separator_count):
              i=answer.find(separator,i)
              if(stop_at_max):
                index.append(i+len(separator))
              else:
                index.append(i)
              i+=len(separator)

      index.sort()
      index.append(len(answer))
      #print(index)
      if(count>=max_limited_chars):
          #print(index[max_limited_chars])
          if(stop_at_max):
            answer=answer[0:index[max_limited_chars]]
          else:
            answer=answer[0:index[max_limited_chars+1]]
          if(verbose):
            print("OUTPUT STOPPED at:",limit_chars)

      return answer

assistant_token = '<|assistant|>'
prompter_token = '<|prompter|>'
end_token = '<|endoftext|>'
context_token= '<|emotion|>'

def predict_emotion(prompt, num_return_emotions=10,include_neutral=True,uncertainty=2,positive_negative_check=3,do_sample=True,verbose=False,recursion=0):
  model_input=prompt
  #num_return_emotions=10
  sequences = generation_pipeline(
    model_input,
    #min_length=10,
    #max_length=200,
    min_new_tokens=2,
    max_new_tokens=5,
    do_sample=do_sample,
    top_k=5,
    num_return_sequences=num_return_emotions,
    eos_token_id=tokenizer.eos_token_id,
  )
  emotion=''
  emotions=[]
  emotions_count={}
  emotion_value=0

  new_emotions=[] #rare event but can happen

  for seq in sequences:
    emotion=seq['generated_text'][len(model_input):].strip()

    emotion=emotion.split('<|assistant|>',1)[0]
    emotion=emotion.split('<|endoftext|>',1)[0]
    emotion = filter_limit_chars(emotion,['|','<','>',',','.'],0,False,False).strip()

    if emotion in emotions_count:
        emotions_count[emotion] += 1
    else:
        emotions_count[emotion] = 1

    if(emotion not in emotions_dict.keys()):
      new_emotions.append(emotion)


  if(len(new_emotions)==0):#No unexpected emotions
    emotion_value= sum(emotions_count[key] * emotions_dict[key] for key in emotions_count)
    emotion = max(emotions_count, key=emotions_count.get)

    if(num_return_emotions>1 and include_neutral):

      if(emotions_count[emotion]<num_return_emotions//uncertainty):
        if(emotion_value<-num_return_emotions//positive_negative_check):
          emotion='negative'
        elif(emotion_value>num_return_emotions//positive_negative_check):
          emotion='positive'
        else:
          emotion = 'neutral'
  
  #  emotion,emotions_count,emotion_value=predict_emotion(prompt, num_return_emotions,include_neutral,uncertainty,positive_negative_check,do_sample,verbose)
  else:#avoid problems if new emotions predicted, try again -> infinite recursion loop possible? but extremely unlikely-> ok it happens with 'cold'
    recursion+=1
    if(recursion<10):
      emotion,emotions_count,emotion_value=predict_emotion(prompt, num_return_emotions,include_neutral,uncertainty,positive_negative_check,do_sample,verbose,recursion)
    else:
      #remove extra emotions from output and return neutral, stops recursion
      emotions_count = {key: value for key, value in emotions_count.items() if key not in new_emotions}
      emotion = 'neutral'
      emotions_value = 0
  #print("\033[94m Emotion:", answer)

  return emotion,emotions_count,emotion_value

emo_files = [f.split('/')[-1] for f in emotional_files]

import json
import os
from google.colab import files

for file_ in json_files:
    if file_.split('/')[-1] not in emo_files:
      print(file_)
      emotions_dict = {
          'surprised':0,
          'excited':0,
          'angry':0,
          'proud':0,
          'sad':0,
          'annoyed':0,
          'grateful':0,
          'lonely':0,
          'afraid':0,
          'terrified':0,
          'guilty':0,
          'impressed':0,
          'disgusted':0,
          'hopeful':0,
          'confident':0,
          'furious':0,
          'anxious':0,
          'anticipating':0,
          'joyful':0,
          'nostalgic':0,
          'disappointed':0,
          'prepared':0,
          'jealous':0,
          'content':0,
          'devastated':0,
          'embarrassed':0,
          'caring':0,
          'sentimental':0,
          'trusting':0,
          'ashamed':0,
          'apprehensive':0,
          'faithful':0,
      }

      with open(file_, 'r', encoding='utf-8') as f:
          final_diz = json.load(f)

      for post in final_diz['evidences']:
          emotion, emotions_count, emotion_value = predict_emotion('<|prompter|>' + post + '<|endoftext|><|emotion|>', num_return_emotions=10, include_neutral=False, uncertainty=2, positive_negative_check=3, do_sample=True, verbose=False, recursion=0)
          for emotion, count in emotions_count.items():
            if emotion in emotions_dict:
              emotions_dict[emotion] += count

      output_filename = '/content/' + file_.split('/')[-1]
      with open(output_filename, 'w', encoding='utf-8') as output_file:
          json.dump(emotions_dict, output_file, indent=2, ensure_ascii=False)

