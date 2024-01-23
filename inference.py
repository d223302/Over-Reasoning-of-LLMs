#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import openai
import time
from google.api_core import retry
import google.generativeai as palm
import requests
from urllib.request import Request, urlopen

openai.api_key = ""
palm_api_key = ''
claude_api_key = ""

# TODO: add more number of generated responses?

def call_claude(prompt, model):
  prompt = prompt.replace('\\n', '\n')
  while True:
#    try:
      data = {
          "max_tokens_to_sample": 512,
          "model": model,
          "prompt": prompt,
          'temperature': 0.7,
      }
      response = requests.post(
        url = "https://api.anthropic.com/v1/complete",
        headers = {
          'accept': 'application/json',
          'anthropic-version': '2023-06-01', 
          'content-type': 'application/json',
          'x-api-key': claude_api_key,
        },
        json = data,
      )
      if response.status_code == 200:
        break
#    except NotImplementedError:
#      print("What happen?") 
  print(response.text)
  response = json.loads(response.text)["completion"]
  return response

 
def call_gpt(cur_prompt, n_reasoning_paths, model):
  print(cur_prompt)
  while True:
    try:
      responses = openai.ChatCompletion.create(
        model = model,
        max_tokens = 512,
        messages = [
          {'role': "system", "content": "You are a helpful assistant. You need to answer the questions of the user accurately."},
          {'role': "user", "content": cur_prompt},
        ],
        #stop = stop,
        temperature = 0.7,
        n = n_reasoning_paths,
      )
      break
    except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:
      time.sleep(0.5)
      pass
  answers = []
  n_input = []
  n_output = []
  for i in range(n_reasoning_paths):
    answers.append(responses['choices'][i]['message']['content'])
  n_input = responses['usage']['prompt_tokens']
  n_output = responses['usage']['completion_tokens']

  return answers, (n_input, n_output)

@retry.Retry()
def call_palm(prompt, candidate_count, model):
  completion = palm.generate_text(
    model = model,
    prompt = prompt,
    temperature = 0.7,
    candidate_count =  candidate_count,
    max_output_tokens = 512,
  )
  answers = [c['output'] for c in completion.candidates]
  return answers

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_set", help = "Path to the json file")
  parser.add_argument("--model", help = "The model name on huggingface")
  parser.add_argument("--output_path", help = "The output path to save the model output")
  parser.add_argument("--question_prefix", default = "", help = "The prefix for question. For example, [INST] in LLaMA2")
  parser.add_argument("--answer_prefix", default = "", help = "The answer template for the model")
  parser.add_argument("--n_output", default = 1, type = int, help = "Number of samples from LLM")
  args = parser.parse_args()

  with open(args.test_set, 'r') as f:
    lines = f.readlines()

  args.question_prefix = args.question_prefix.replace('\\n', '\n')

  output_path = os.path.join(args.output_path, args.test_set.split("/")[-1].split('.')[0])
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  if "gpt" in args.model or "text-" in args.model:
    use_api = True
  elif 'claude' in args.model:
    use_api = True
  else:
    use_api = False

  if not use_api:
    model = AutoModelForCausalLM.from_pretrained(
      args.model, 
      trust_remote_code = True,
      device_map = "auto",
      low_cpu_mem_usage = True,
      torch_dtype = torch.float16,
      load_in_8bit = True,
    )
    model.eval()
  #  model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
  
      
    generation_config = GenerationConfig(
        max_new_tokens = 512,
        do_sample = True,
        temperature = 0.7,
        num_return_sequences = args.n_output,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
    )
  if "bison" in args.model:
    palm.configure(api_key = palm_api_key)


  with open(os.path.join(output_path, args.model.strip('/').split('/')[-1] + '.json'), 'w') as f:
    for line in lines:
      instance = json.loads(line.strip())
      q = instance['question']
      a = instance['answer']
      prompt = args.question_prefix.replace("QUESTION", q)
      if not use_api:
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids.to("cuda"),  generation_config = generation_config)
        pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
      else:
        if 'bison' in args.model: # Google models
          pred = call_palm(prompt, args.n_output, args.model)
        elif 'claude' in args.model:
          pred = call_claude(prompt, args.model)
        elif 'gpt' in args.model or 'text' in args.model: # OpenAI models
          pred, (_, _) = call_gpt(prompt, args.n_output, args.model)
          #pred = pred[0]
        else:
          raise NotImplementedError

      print(f"Ground truth: {a}")
      print(f"Prediction: {pred}")
      print("=" * 20 + '\n')
      result = {
        "ground truth": a,
        "prediction": pred,
      }
      f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
  main()
