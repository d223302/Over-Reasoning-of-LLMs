#!/usr/bin/env python3
import numpy as np
import json

def get_verdict(p, mode):
  p = p['response'][0]
  verdict = p.split('[[')[-1].split(']]')[0]
  if verdict == "C":
    verdict = "T"
    return verdict


  if mode == 0:
    if verdict == "A":
      verdict = "L"
    else:
      verdict = "S"
  else:
    if verdict == "A":
      verdict = "S"
    else:
      verdict = "L"
  return verdict

with open("predictions/geval/gpt4_eval_accurate_result.json", 'r') as f:
  eval_result = json.load(f)

with open("predictions/geval/human_eval_accurate.json", 'r') as f:
  human = json.load(f)

print(len(eval_result))
print(len(human))

llama_result = []
chatgpt_result = []

for idx in range(125):
  if human[idx * 2].strip() == "1":
    verdict_llama_1 = eval_result[4 * idx]
    verdict_llama_2 = eval_result[4 * idx + 1]

    llama_result.append(get_verdict(verdict_llama_1, 0))
    llama_result.append(get_verdict(verdict_llama_2, 1))

  if human[idx * 2 + 1].strip() == "1":
    verdict_chatgpt_1 = eval_result[4 * idx + 2]
    verdict_chatgpt_2 = eval_result[4 * idx + 3]
    chatgpt_result.append(get_verdict(verdict_chatgpt_1, 0))
    chatgpt_result.append(get_verdict(verdict_chatgpt_2, 1))


with open("predictions/geval/gpt4_eval_accurate_result.json", 'r') as f:
  eval_result = json.load(f)

with open("predictions/geval/human_eval_accurate.json", 'r') as f:
  human = json.load(f)

print(len(eval_result))
print(len(human))


for idx in range(125):
  if human[idx * 2].strip() == "1":
    verdict_llama_1 = eval_result[4 * idx]
    verdict_llama_2 = eval_result[4 * idx + 1]

    llama_result.append(get_verdict(verdict_llama_1, 0))
    llama_result.append(get_verdict(verdict_llama_2, 1))

  if human[idx * 2 + 1].strip() == "1":
    verdict_chatgpt_1 = eval_result[4 * idx + 2]
    verdict_chatgpt_2 = eval_result[4 * idx + 3]
    chatgpt_result.append(get_verdict(verdict_chatgpt_1, 0))
    chatgpt_result.append(get_verdict(verdict_chatgpt_2, 1))



for result in [llama_result, chatgpt_result]:
  #print(result)
  print(f"Total samples: {len(result)}")
  print(f"Longer is better : {result.count('L') / len(result)}")
  print(f"Shorter is better: {result.count('S') / len(result)}")
  print(f"Tie              : {result.count('T') / len(result)}")
