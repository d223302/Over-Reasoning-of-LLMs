#!/usr/bin/env python3
import string
import re
import openai
import urllib.request, json 
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
from nltk.tokenize import sent_tokenize
import random
import time

random.seed(31616)
openai.api_key = ""

template = '''Your task is to convert a declarative sentence into a question and the answer to that question should be a number. Importantly, the answer (number) to the question should already be included in the original sentence. If the answer need to be obtained by calculation, the question is invalid. Even simple calculation is not allowed.  Keep the question as simple as possible. For example:
Example 1:
Original sentence: Alyssa, Keely, and Kendall ordered 100 chicken nuggets from a fast-food restaurant.
Answer (number only): 100
Question: How many chicken nuggets did Alyssa, Keely, and Kendall order?
Explanation: The number 100 already appeared in the original sentence, so the question fulfill the requirements.

Example 2: 
Original sentece: Lilah's family gallery has 400 photos.
Answer (number only): 400
Question: How many photos are there in Lilah's family gallery?
Explanation: The number 400 already appeared in the original sentence, so the question fulfill the requirements.

Example 3: 
Original sentence: {ORIG_SENT}
Answer (number only): {ANS}
Question: 
'''


def call_gpt(cur_prompt):
  print(cur_prompt)
  while True:
    try:
      responses = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        max_tokens = 100,
        messages = [
          {'role': "system", "content": "You are a helpful assistant. You need to answer the questions of the user accurately. You need to strictly follow the instructions."},
          {'role': "user", "content": cur_prompt},
        ],
        #stop = stop,
        temperature = 0,
        n = 1,
      )
      #print(json.dumps(responses, indent = 2))
      break
    except (openai.error.APIError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout) as e:
      time.sleep(2)
  answers = []
  n_input = []
  n_output = []
  for i in range(len(responses['choices'])):
    answers.append(responses['choices'][i]['message']['content'])
  n_input = responses['usage']['prompt_tokens']
  n_output = responses['usage']['completion_tokens']

  return answers, (n_input, n_output)


dataset = load_dataset("gsm8k", 'main', split = "test")
if not os.path.exists('generated_dataset'):
  os.makedirs('generated_dataset')

#dataset = dataset.select([i for i in range(80, len(dataset))])

count = 0 
with open('generated_dataset/trivial_gsm8k_random_test.json', 'w') as f:
  for instance in tqdm(dataset):
    orig_q = instance['question']
    sentences = sent_tokenize(orig_q)
    sent_with_num = []

    ## Only consider questions with more than two numbers
    if len(re.findall(r'-?\d+\.?\d*', " ".join(sentences[:-1]))) < 2:
      continue

    for sentence in sentences:
      num = re.findall(r'-?\d+\.?\d*', sentence)
      if len(num) > 0 and sentence.strip()[-1] != "?":
        sent_with_num.append(sentence)
    if len(sent_with_num) == 0:
      continue
    random_sentence = random.choice(sent_with_num)
    potential_ground_truth = re.findall(r'-?\d+\.?\d*', random_sentence)[0]
    prompt = template.replace("{ORIG_SENT}", random_sentence).replace("{ANS}", potential_ground_truth)
    answers, _ = call_gpt(prompt)
    new_question = answers[0].split("Question:")[-1].split('\n')[0].strip()
    print(new_question)
    print(potential_ground_truth)
    sentences.pop()
    sentences.append(new_question)
    trivial_question = " ".join(sentences)
    f.write(
      json.dumps(
        {
        "question": trivial_question,
        "answer": float(potential_ground_truth),
        }
      ) + '\n'
    )
    count += 1
    if count == 3500:
      break

