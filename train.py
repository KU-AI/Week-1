import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Resnet18
from load_data import create_dataset
from torch.utils.tensorboard import SummaryWriter

def train(epochs=80, lr=0.01, weight_decay=5e-4, optimizer='SGD',
          save_path='model_state_dict.pt'):
    assert optimizer == 'SGD' or optimizer == 'Adam', 'Not supporting optimizer'

    model = Resnet18()
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    train_total = len(train_set.dataset)
    running_loss = 0.0

    check_print = np.floor((2000 / train_set.batch_size) / 3)

    for epoch in range(epochs):
        # train
        train_correct = 0.0
        model.train()
        for i, data in enumerate(train_set):
            img, label = data
            img, label = img.cuda(), label.cuda()

            output = model(img)
            train_loss = criterion(output, label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_correct += (output.max(dim=1)[1] == label).sum().float()
            running_loss += train_loss.item()
            if (i + 1) % check_print == 0:
                writer.add_scalar('training_loss', running_loss / check_print,
                                  (epoch + 1) * len(train_set) + i)
                print('[%d/%3d, %3d/%3d] loss: %.3f' % (epoch + 1, epochs, i + 1,
                                                        len(train_set), running_loss / check_print))
                running_loss = 0.0

        # val
        with torch.no_grad():
            val_correct = 0.0
            val_total = len(val_set.dataset)
            for data in val_set:
                val_img, val_label = data
                val_img, val_label = val_img.cuda(), val_label.cuda()

                val_output = model(val_img)
                val_loss = criterion(val_output, val_label)
                val_correct += (val_output.max(dim=1)[1] == val_label).sum().float()

            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100

        writer.add_scalar('loss/train_loss', train_loss, epoch + 1)
        writer.add_scalar('loss/val_loss', val_loss, epoch + 1)
        writer.add_scalar('acc/train_acc', train_acc, epoch + 1)
        writer.add_scalar('acc/val_acc', val_acc, epoch + 1)

        print('epoch: %3d/%3d, train_loss: %.4f, train_acc: %.2f%%, val_acc: %.2f%%'
              % (epoch + 1, epochs, train_loss.item(), train_acc, val_acc))

    torch.save(model.state_dict(), save_path)
    print('End Learning')

def test(load_path='model_state_dict.pt'):
    model = Resnet18()
    model.load_state_dict(torch.load(load_path))

    if torch.cuda.is_available():
        model.cuda()

    correct = 0.0
    total = len(test_set.dataset)

    with torch.no_grad():
        for data in test_set:
            img, label = data
            img, label = img.cuda(), label.cuda()

            y_pred = model(img)
            correct += (y_pred.max(dim=1)[1] == label).sum().float()
        acc = correct / total * 100
    print('test acc: %2f' % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=80,
                        help='set training epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01,
                        help='set learning rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4,
                        help='set weight decay')
    parser.add_argument('-o', '--optimizer', type=str, default='SGD',
                        help='set optimizer')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='set batch size')
    parser.add_argument('-dp', '--data_path', type=str, default='data/',
                        help='set data path')
    parser.add_argument('-s', '--shuffle', type=bool, default=True,
                        help='Shuffle or Not')
    parser.add_argument('-sp', '--save_path', type=str,
                        default='model_state_dict.pt',
                        help='Name where model save')

    args = parser.parse_args()

    train_set, val_set, test_set = create_dataset(args.batch_size, args.data_path,
                                                  args.shuffle)
    train(args.epochs, args.learning_rate, args.weight_decay, args.optimizer,
          args.save_path)
    test(args.save_path)
