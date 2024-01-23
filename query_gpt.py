#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_set", help = "Path to the json file")
  parser.add_argument("--model", help = "The model name on huggingface")
  parser.add_argument("--output_path", help = "The output path to save the model output")
  parser.add_argument("--question_prefix", default = "", help = "The prefix for question. For example, [INST] in LLaMA2")
  parser.add_argument("--answer_prefix", default = "", help = "The answer template for the model")
  args = parser.parse_args()

  with open(args.test_set, 'r') as f:
    lines = f.readlines()

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

  output_path = os.path.join(args.output_path, args.test_set.split("/")[-1].split('.')[0])
  if not os.path.exists(output_path):
    os.makedirs(output_path)
    
  generation_config = GenerationConfig(
      max_new_tokens = 512,
      do_sample = True,
      temperature = 1.0,
      pad_token_id = tokenizer.pad_token_id,
      eos_token_id = tokenizer.eos_token_id,
  )


  with open(os.path.join(output_path, args.model.strip('/').split('/')[-1] + '.json'), 'w') as f:
    for line in lines:
      instance = json.loads(line.strip())
      q = instance['question']
      a = instance['answer']
      prompt = args.question_prefix.replace("QUESTION", q)
      inputs = tokenizer(prompt, return_tensors="pt")
      generate_ids = model.generate(inputs.input_ids.to("cuda"),  generation_config = generation_config)
      pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]      
      print(pred)
      result = {
        "ground truth": a,
        "prediction": pred,
      }
      f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
  main()
