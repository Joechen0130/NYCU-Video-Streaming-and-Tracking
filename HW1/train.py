# import your model from net.py
from net import my_network
import torch
import pandas as pd
import os
from dataloader import Dataset,train_transform,test_transform
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #hyper-parameter
    batch_size = 16
    lr = 1e-4
    epochs = 200

    #load data
    df = pd.read_csv('./dataset/train/train.csv')
    image_path_list = []
    label_list = df["label"].to_list()
    for e in df["name"].to_list():
        # print(os.path.join(f'./dataset/train',e))
        image_path_list.append(os.path.join(f'./dataset/train', e))

    train_data, val_data, train_label, val_label = train_test_split(image_path_list, label_list,
                                                                      test_size=0.2, stratify=label_list,
                                                                      random_state=42)

    train_data = Dataset(train_data, train_label, transform=train_transform)
    val_data = Dataset(val_data, val_label, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    model = my_network()#load model
    #print(model)
    model = model.to(device)

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # pass in the parameters to be updated and learning rate

    train_loss_list = []
    test_loss_list = []
    train_acc_list, test_acc_list = [], []

    # Training Loop
    print("Training Start:")
    for epoch in range(epochs):
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))
        model.train()  # start to train the model, activate training behavior
        iter = 0
        train_loss = 0
        test_loss = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            # reshape images
            optimizer.zero_grad()
            #print(images.shape, labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()  # run back propagation
            optimizer.step()  # optimizer update all model parameters
            # loss
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum()

            iter += 1

        print("loss: %.3f train_acc: %.3f " % (train_loss/iter, correct_train/total_train),end="")

        # --------------------------
        # Testing Stage
        # --------------------------
        model.eval()
        iter = 0
        for i, (images, labels) in enumerate(val_loader):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum()
                iter += 1
        print("tes_loss: %.3f test_acc: %.3f" % (test_loss/iter, correct_test/total_test))

        train_loss_list.append(train_loss/iter)
        test_loss_list.append(test_loss/iter)
        train_acc_list.append((correct_train/total_train).cpu())
        test_acc_list.append((correct_test/total_test).cpu())

    torch.save(model.state_dict(), "./w_311554014.pth")
    #torch.save(model,"./w_311554014.pth")

    fig_dir = './fig/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    plt.figure()
    plt.plot(list(range(epochs)), train_loss_list)
    plt.plot(list(range(epochs)), test_loss_list)
    plt.title('Training Loss')
    plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend(['train_loss','test_loss'], loc='upper left')
    plt.savefig(os.path.join(fig_dir, 'loss.png'))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(list(range(epochs)), train_acc_list)  # plot your training accuracy
    plt.plot(list(range(epochs)), test_acc_list)  # plot your testing accuracy
    plt.title('Training acc')
    plt.ylabel('acc (%)'), plt.xlabel('epoch')
    plt.legend(['training acc', 'testing acc'], loc='upper left')
    plt.savefig(os.path.join(fig_dir, 'acc.png'))
    plt.show()

if __name__ == "__main__":
    train()