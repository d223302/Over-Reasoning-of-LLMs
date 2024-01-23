#!/usr/bin/env python3
import numpy as np
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

PALM_KEY = ""

def main():
  args = argparse.ArgumentParser()
  parser.add_argument("--test_set", help = "Path to the json file")
  parser.add_argument("--model", help = "The model name on huggingface")
  parser.add_argument("--question_prefix", default = "", help = "The prefix for question. For example, [INST] in LLaMA2")
  parser.add_argument("--answer_prefix", default = "", help = "The answer template for the model")
  args = argparse.parse_args()

  with open(args.test_set, 'r') as f:
    lines = f.readline()

  model = AutoModelForCausalLM.from_pretrained(args.model)
  tokenizer = AutoTokenizer.from_pretrained(args.model)

  for line in lines:
    instance = json.loads(line.strip())
    q = instance['question']
    a = instance['answer']
    prompt = args.question_prefix.replace("QUESTION", q)
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name == "__main__":
  main()
