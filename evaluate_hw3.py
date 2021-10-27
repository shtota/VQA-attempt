import torch
import torch.utils.data as data
from data_utils import get_vocab, VqaDataset, get_datasets
from models import VqaModel
import os
from parameters import DEVICE, BATCH_SIZE, N_WORKERS, PREPROCESSED_FILE_PATH
from preprocess_images import preprocess_images


def evaluate_hw3():
    print('starting. device:', DEVICE)
    if not os.path.exists(PREPROCESSED_FILE_PATH):
        print('No preprocessed images found. Preprocessing. It will take ~hour')
        preprocess_images()
    vocabs = get_vocab()
    train_data, val_data = get_datasets(vocabs)
    val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    input_text_size = train_data.question_vocab_size
    output_text_size = train_data.answer_vocab_size
    print('Loading model')
    model = VqaModel(input_text_size, output_text_size)
    model.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc: storage))
    model.to(DEVICE)
    model.eval()
    total_error = 0
    answers = {}
    print('Starting evaluation')

    batch_idx = 0
    for batch_sample in val_loader:
        images = batch_sample['image'].to(DEVICE)
        questions = batch_sample['question'].to(DEVICE)
        labels = batch_sample['answer'].to(DEVICE)
        items = batch_sample['item_no']
        with torch.set_grad_enabled(False):
            outputs = model(images, questions)
            _, predicted = torch.max(outputs.data, 1)

            num_answered = (predicted == labels).sum().item()
            error = 1 - num_answered / BATCH_SIZE
            total_error += error
            for i, item in enumerate(items):
                answers[item.item()] = predicted[i].item()
        batch_idx += 1
        del items
        del images
    average_error = total_error/(len(answers)/128)
    print('0-1 accuracy of multi-choice winner on validation set is:', 1-average_error)

    total_acc = 0
    for item, answer_list in enumerate(val_data.answers):
    #for item in answers.keys():
    #    answer_list = val_data.answers[item]
        my_answer = answers[item]
        count = answer_list.count(my_answer)
        accuracy = min(1, count/3)
        total_acc += accuracy

    print('VQA accuracy of all answers(https://visualqa.org/evaluation.html) on validataion set is:', total_acc/len(answers))


if __name__ == '__main__':
    evaluate_hw3()