# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import json
import numpy as np
import random
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm
from QA_Dataset import QA_Dataset

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    same_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = (
            os.getenv('LHY_WK7_Hw_data') or
            'data/'
    )
    fp16_training = False

    if fp16_training:
        accelerator = Accelerator(fp16=True)
        device = accelerator.device

    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")


    def evaluate(data, output):
        ##### TODO: Postprocessing #####
        # There is a bug and room for improvement in postprocessing
        # Hint: Open your prediction file to see what is wrong

        answer = ''
        max_prob = float('-inf')
        num_of_windows = data[0].shape[1]

        for k in range(num_of_windows):
            # Obtain answer by choosing the most probable start position / end position
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)

            # Probability of answer is calculated as sum of start_prob and end_prob
            prob = start_prob + end_prob

            # Replace answer if calculated probability is larger than previous windows
            if prob > max_prob:
                max_prob = prob
                # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])

        # Remove spaces in answer (e.g. "大 金" --> "大金")
        return answer.replace(' ', '')

    train_questions, train_paragraphs = read_data(DATA_DIR + "hw7_train.json")
    dev_questions, dev_paragraphs = read_data(DATA_DIR + "hw7_dev.json")
    test_questions, test_paragraphs = read_data(DATA_DIR + "hw7_test.json")

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                          add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                        add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                         add_special_tokens=False)

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

    train_batch_size = 16

    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    num_epoch = 6
    validation = True
    logging_step = 100
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()

    print("Start Training ...")

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            print("-" * 80)
            print("input_id : {}".format(data[0][0]))
            print("-" * 80)
            print("token_type_ids : {}".format(data[1][0]))
            print("-" * 80)
            print("input_id : {}".format(data[2][0]))
            print("-" * 80)
            print("input_id : {}".format(data[3][0]))
            print("-" * 80)
            print("input_id : {}".format(data[4][0]))
            print("-" * 80)
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                           end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze().to(device), token_type_ids=data[1].squeeze().to(device),
                                   attention_mask=data[2].squeeze().to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()

    # Save a model and its configuration file to the directory 「saved_model」
    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
    print("Saving Model ...")
    model_save_dir = "saved_model"
    model.save_pretrained(model_save_dir)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
