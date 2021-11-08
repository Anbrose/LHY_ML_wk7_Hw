import json
from QA_Dataset import QA_Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast




model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("data/hw7_train.json")
question = train_questions[3]
tokenized_paragraph = tokenizer(train_paragraphs[question['paragraph_id']])

answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])



print("-" * 80)
print(answer_start_token)
print("-" * 80)
print(answer_end_token)
print("-" * 80)
print(tokenized_paragraph)
print("-" * 80)
print(train_paragraphs[question['paragraph_id']])
print("-" * 80)
