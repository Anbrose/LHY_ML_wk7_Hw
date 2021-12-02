import torch
import numpy as np
import random
import json
import tqdm
from transformers import BertForQuestionAnswering, BertTokenizerFast
from Chinese_QA_Dataset import QA_Dataset
from torch.utils.data import DataLoader, Dataset

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def read_QA(path):
    f = open(path, 'r', encoding="UTF-8")
    data = json.load(f)
    return data["questions"], data["paragraphs"]


if __name__ == '__main__':
    train_questions, train_paragraphs = read_QA("data/hw7_train.json")
    # test_questions,test_paragraphs = read_QA("data/hw7_test.json")
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    #
    # tokenized_train_paragraphs = tokenizer(train_paragraphs, add_special_tokens=False)
    # tokenized_test_paragraphs = tokenizer()

    dataset_train = QA_Dataset(train_questions, train_paragraphs)

    train_loader = DataLoader(dataset_train,batch_size=1,shuffle=False)

    for data in train_loader:
        question_text, paragraph, answer_start, answer_end = data
        inputs = tokenizer(question_text, paragraph)
        #print(inputs['input_ids'][0])
        break
        # output = model(**inputs)
        # answer_start = torch.argmax(output.start_logits)
        # answer_end = torch.argmax(output.end_logits) + 1
        #
        # answer = tokenizer.decode()
        # break

    # epoch = 6
    #
    # for i in range(epoch):


    #model.train()

    #(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None)



