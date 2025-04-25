# import your model from net.py
from net import my_network
import torch
import pandas as pd
import os
from dataloader import test_transform
from PIL import Image
import glob
from sklearn.metrics import classification_report
'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''

def test():
    device = torch.device("cpu")
    print(device)
    # load data
    test_path = "./dataset/test"
    test_list = []
    transform = test_transform
    test_path_list = []

    for e in range(len(glob.glob(test_path+"/*"))):
        file = str(e) + ".jpg"
        test_path_list.append(file)

    for e in test_path_list:
        file = os.path.join(test_path,e)
        img = Image.open(file).convert('RGB')
        img = transform(img)
        test_list.append(img)

    model = my_network()  # load model
    model.load_state_dict(torch.load("./w_311554014.pth"))
    #model = torch.load("./w_311554014.pth")
    model = model.to(device)

    # --------------------------
    # Testing Stage
    # --------------------------
    model.eval()
    y_pred = []

    for e in test_list:
        with torch.no_grad():  # 測試階段不需要求梯度
            e = e.to(device)
            e = torch.unsqueeze(e, 0)
            outputs = model(e)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.append(predicted.cpu().item())

    dict = {'name': test_path_list, 'label': y_pred}
    df = pd.DataFrame(dict)
    df.to_csv('./pred_311554014.csv',index=False)


if __name__ == "__main__":
    test()