import torch
from torch.utils.data import DataLoader, Dataset


class QA_Dataset(Dataset):
    def __init__(self, questions, paragraphs):
        self.questions = questions
        self.paragraphs = paragraphs

    def __len__(self):
        return len(self.questions)

    '''
        :returns
            question_text: str
            paragraph_text: str
            answer_start: int
            answer_end: int
    '''
    def __getitem__(self, item):

        question = self.questions[item]
        question_text = question['question_text']
        paragraph = self.paragraphs[question['paragraph_id']]
        answer_start = question['answer_start']
        answer_end = question['answer_end']
        return question_text, paragraph, answer_start, answer_end