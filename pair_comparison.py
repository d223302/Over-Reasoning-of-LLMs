#!/usr/bin/env python3
import json
import openai
import numpy as np
import time
import os

openai.api_key = ""

l_valid = set(json.load(open("predictions/llama7b-chat_inaccurate.json")))
chatgpt_valid = set(json.load(open("predictions/gpt3.5-turbo_inaccurate.json")))

both_valid = l_valid.intersection(chatgpt_valid)

with open("result/default/trivial_gsm8k_random/Llama-2-7b-chat-hf.json", 'r') as f:
  llama_predictions = f.readlines()

with open("result/default/trivial_gsm8k_random/gpt-3.5-turbo-0613.json", 'r') as f:
  chatgpt_predictions = f.readlines()

def get(line):
  data = json.loads(line)
  prediction = data['prediction']
  ground_truth = data['ground truth']
  return prediction.strip(), ground_truth



def call_gpt(cur_prompt, n_reasoning_paths, model):
  print(cur_prompt)
  while True:
    try:
      responses = openai.ChatCompletion.create(
        model = model,
        max_tokens = 512,
        messages = [
          {'role': "system", "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."},
          {'role': "user", "content": cur_prompt},
        ],
        #stop = stop,
        temperature = 0.0,
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


result = []
human_eval = []

start_time = time.time()

for idx, (l, c) in enumerate(zip(llama_predictions, chatgpt_predictions)):
  if not (idx in both_valid):
    continue
  l_prediction, gt = get(l)
  c_prediction, _ = get(c)
  question = l_prediction.split('[/INST]')[0].replace('[INST]', '').strip()
  #print(question)
  l_prediction = l_prediction.split('[/INST]')[-1].strip()


  if float(gt) == int(gt):
    gt = int(gt)
  else:
    gt = float(gt)

  simple_answer = f"The answer is {gt}"
 
  for answer_a in [l_prediction, c_prediction]:
    prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{simple_answer}\n[The End of Assistant B's Answer]"
    reverse_prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{simple_answer}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_a}\n[The End of Assistant B's Answer]"
    
    #start = "\033[1m"
    #end = "\033[0;0m"
    #print(f'{start}Idx{end}: {idx}')
    #print(f"{start}Question{end}:\n{question}")
    #print(f"{start}Answer{end}: {gt}")
    #print(f"{start}Model output{end}:\n{answer_a}")
    #print(f"-" * 20)
    #print(f"Is the model answer correct? Options: \n{start}1{end}: the model answer is correct and matches the ground truth\n{start}2{end}: the model answer is correct but the ground truth is wrong\n{start}3{end}: the model answer is wrong while the ground truth is correct\n{start}4{end}:The question is invalid")
    #verdict = input()
    #human_eval.append(verdict)
    #print('-' * 20)
    #os.system('clear')
    #print(f"\nAverage number of samples per minute: {len(human_eval) / ((time.time() - start_time) / 60.0)}")
    #print(f"{(time.time() - start_time) / 60.0} minutes have passed and you labeled {len(human_eval)}/250 instances")
    #print(f"Invalid: {human_eval.count(4)}/250")
    #print("==" * 10)


    response, _ = call_gpt(
      prompt,
      1,
      "gpt-4-0613",
    )

    result.append(
      {
        "answer_a": answer_a,
        "answer_b": simple_answer,
        "response": response,
      }
    )

    response, _ = call_gpt(
      reverse_prompt,
      1,
      "gpt-4-0613",
    ) 

    result.append(
      {
        "answer_b": answer_a,
        "answer_a": simple_answer,
        "response": response,
      }
    )

    with open('gpt4_eval_inaccurate_result.json', 'w') as f:
      json.dump(result, f, indent = 4)
    #with open('human_eval_inaccurate.json', 'w') as f:
    #  json.dump(human_eval, f, indent = 4)

  if len(result) > 500:
    break


