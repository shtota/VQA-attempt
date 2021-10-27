import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
import pickle
import json
import time
from data_utils import get_vocab, VqaDataset, get_datasets
from preprocess_images import preprocess_images
from models import VqaModel
from parameters import DEVICE, BATCH_SIZE, N_WORKERS, PREPROCESSED_FILE_PATH

LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.1
START_EPOCH = 0
NUM_EPOCHS = 25


def map_agreement(x):
    return min(1, 0.3*x)


def main():
    start_epoch = START_EPOCH
    os.makedirs('results', exist_ok=True)
    print('starting. device:', DEVICE)

    if not os.path.exists(PREPROCESSED_FILE_PATH):
        print('No preprocessed images found. Preprocessing. It will take ~hour')
        preprocess_images()

    vocabs = get_vocab()
    train_data, val_data = get_datasets(vocabs)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)

    print('Creating model')
    input_text_size = train_data.question_vocab_size
    output_text_size = train_data.answer_vocab_size
    model = VqaModel(input_text_size, output_text_size)

    if start_epoch:
        if os.path.exists('model'+str(start_epoch-1)+'.pkl'):
            model.load_state_dict(torch.load('model'+str(start_epoch-1)+'.pkl', map_location=lambda storage, loc: storage))
            print('Loaded model state from epoch ', start_epoch-1)
        else:
            start_epoch = 0
    model.to(DEVICE)
    params = list(model.parameters())
    print('Num of trainable parameters:', sum(p.numel() for p in params if p.requires_grad))
    params = list(model.attention.parameters()) + list(model.question_encoder.parameters()) \
             + list(model.linear1.parameters()) + list(model.linear2.parameters())

    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print('training', len(train_loader), 'batches')
    start = time.time()
    results = []

    for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        batch_idx = 0
        total_loss = 0
        total_error = 0
        for batch_sample in train_loader:
            images = batch_sample['image'].to(DEVICE)
            questions = batch_sample['question'].to(DEVICE)
            labels = batch_sample['answer'].to(DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(images, questions)  # [batch_size, ans size]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                num_answered = (predicted == labels).sum().item()
                error = 1 - num_answered / BATCH_SIZE
                total_error += error
            if batch_idx % 1000 == 0:
                print('epoch', epoch, time.time()-epoch_start, 'training',batch_idx,'batch_loss', loss.item(), 'last_error', error)
            batch_idx += 1
        results.append(['train', total_loss / len(train_loader), total_error / len(train_loader)])
        with open('results/train_'+str(epoch)+'_' + str(total_loss / len(train_loader))[:5] + '_'
                  + str(total_error / len(train_loader))[:5], 'w') as f:
            f.write('')
        print('finished training. Average epoch stats:', results[-1])
        torch.save(model.state_dict(), 'model'+str(epoch)+'.pkl')
        scheduler.step()

        # evaluate on train/val datasets both loss and error
        model.eval()

        total_loss = 0
        total_error = 0
        batch_idx = 0
        for batch_sample in val_loader:
            images = batch_sample['image'].to(DEVICE)
            questions = batch_sample['question'].to(DEVICE)
            labels = batch_sample['answer'].to(DEVICE)
            with torch.set_grad_enabled(False):
                outputs = model(images, questions)  # [batch_size, ans size]
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                num_answered = (predicted == labels).sum().item()
                error = 1 - num_answered/BATCH_SIZE
                total_error += error
            batch_idx += 1
            if batch_idx % 500 == 0:
                print(time.time()-epoch_start, 'testing on val',batch_idx, 'epoch', epoch, 'last loss', loss.item(), 'last error', error)
        results.append(['val', total_loss/len(val_loader), total_error/len(val_loader)])
        with open('results/val_'+str(epoch)+'_' + str(total_loss / len(val_loader))[:6] + '_'
                  + str(total_error / len(val_loader))[:6], 'w') as f:
            f.write('')
        print('epoch took', time.time()-epoch_start, results[-1])
    print('total runtime', time.time()-start)

    with open('results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
