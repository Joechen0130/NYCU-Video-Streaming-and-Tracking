# import your model from net.py
from net import my_network
import torch
import pandas as pd
import os
from dataloader import Dataset,train_transform,test_transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''

def test():

    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    df = pd.read_csv('./dataset/train/train.csv')
    image_path_list = []
    label_list = df["label"].to_list()
    for e in df["name"].to_list():
        image_path_list.append(os.path.join(f'./dataset/train', e))

    train_data, val_data, train_label, val_label = train_test_split(image_path_list, label_list,
                                                                    test_size=0.2, stratify=label_list,
                                                                    random_state=42)

    train_data = Dataset(train_data, train_label, transform=train_transform)
    val_data = Dataset(val_data, val_label, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    model = my_network()  # load model
    model.load_state_dict(torch.load("./w_311554014.pth"))
    #model = torch.load("./w_311554014.pth")
    model = model.to(device)

    # --------------------------
    # Testing Stage
    # --------------------------
    model.eval()
    y_true = []
    y_pred = []

    for i, (images, labels) in enumerate(val_loader):
        with torch.no_grad():  # 測試階段不需要求梯度
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    target_names = ['class 0', 'class 1', 'class 2',
                    'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8',
                    'class 9', 'class 10', 'class 11',]

    print(classification_report(y_true, y_pred, target_names=target_names))

    def test():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load data
        test_path = "./dataset/test"
        test_list = []
        transform = test_transform
        for e in range(len(glob.glob(test_path + "/*"))):
            print(os.path.join(test_path, str(e) + ".jpg"))
            file = os.path.join(test_path, str(e) + ".jpg")
            img = Image.open(file).convert('RGB')
            img = transform(img)
            test_list.append(img)

        model = my_network()  # load model
        model.load_state_dict(torch.load("./w_311554014.pth"))
        # model = torch.load("./w_311554014.pth")
        model = model.to(device)

        # --------------------------
        # Testing Stage
        # --------------------------
        model.eval()
        y_pred = []

        df = pd.read_csv('./GT.csv')
        y_true = df["label"].to_list()

        for e in test_list:
            with torch.no_grad():  # 測試階段不需要求梯度
                e = e.to(device)
                e = torch.unsqueeze(e, 0)
                outputs = model(e)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted.cpu().item())
                # y_true.extend(labels.cpu().numpy())

        target_names = ['class 0', 'class 1', 'class 2',
                        'class 3', 'class 4', 'class 5',
                        'class 6', 'class 7', 'class 8',
                        'class 9', 'class 10', 'class 11', ]
        print(y_pred)
        print(y_true)
        print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    test()