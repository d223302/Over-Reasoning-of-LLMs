# Over-Reasoning and Redundant Calculation of Large Language Models

This repo contains the codes and dataset used in our EACL 2024 [paper](https://arxiv.org/abs/2401.11467): "Over-Reasoning and Redundant Calculation of Large Language Models".

### GSM8K-Zero Dataset
The GSM8k-Zero dataset we constructed is the `gsm8k_zero.json`.
There are 2978 questions and their corresponding answers in the json file.
We also share the dataset on 🤗 Hugging Face Datasets with the name [dcml0714/GSM8K-Zero](https://huggingface.co/datasets/dcml0714/GSM8K-Zero).
You can load GSM8K-Zero with Hugging Face Datasets by
```
from datasets import load_dataset
dataset = load_dataset("dcml0714/GSM8K-Zero", split = "test")
```



### The result of prompting LLMs used in our paper
The results of prompting LLMs can be founded in `result/default/trivial_gsm8j_random`.
The results we provide here are unfiltered; for each json file, there should be 3,500 lines.
As mentioned by our paper, we use the results of GPT-4 to filter the questions.
We provide the index (`gpt4_valid_questions.json`) file of the questions that are selected included in `gsm8k_zero.json`.
If you want to extract the results on GSM8K-Zero, from the `.json` files in `result`, you just need to load the result `.json` file, each line as an element in a list, and select the elements based on the indices in `gpt4_valid_questions.json`.

### Disclaimer
we rely on ChatGPT and GPT-4 to construct GSM8K-Zero, so noises in the constructed dataset are inevitable.
We emphasize that future researchers need to keep the noises in the dataset in mind and take special caution when interpreting the results evaluated on GSM8K-Zero.
To understand the noises in the dataset, the authors randomly selected 250 samples from GSM8K-Zero and reviewed them.
As stated in Section 2 in our paper, we estimate that 85% of question-answer pairs in GSM8K-Zero are valid.
We present the details about our manual review of the dataset in the appendix of our paper.
We also discuss that our results and observations in the main content still hold when considering the noises in the dataset.


### Cite
If you use our dataset or find our dataset useful, please cite our paper using
```
@misc{chiang2024overreasoning,
    title={Over-Reasoning and Redundant Calculation of Large Language Models},
    author={Cheng-Han Chiang and Hung-yi Lee},
    year={2024},
    eprint={2401.11467},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Contact
If you have any questions about our paper or GSM8K-Zero, please contact the first author by e-mail or simply open an issue here.

