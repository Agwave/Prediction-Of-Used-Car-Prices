import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# model
class UsedCarPred(nn.Module):

    def __init__(self, input_size):
        super(UsedCarPred, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, data):
        out = self.net(data).view(-1)
        return out

# data
def get_data(path='Train_data.h5'):
    Train_data = pd.read_hdf(path)
    Y_data = Train_data.price.values
    X_data = Train_data.drop('price', axis=1).values

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_data)
    x_data = min_max_scaler.transform(X_data)
    print('X_data: {}'.format(x_data.shape))
    print('Y_data: {}'.format(Y_data.shape))
    return x_data, Y_data

def get_test_data(path='Test_data.h5'):
    Test_data = pd.read_hdf(path)
    X_test = Test_data.drop('SaleID', axis=1).values
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_test)
    x_test = min_max_scaler.transform(X_test)
    print('X_test: {}'.format(X_test.shape))
    return x_test

# 训练和预测
def make_train_step(model, criterion, optimizer):
    def train_step(input, pred):
        model.train()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, pred)
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

def criterion(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))

def train_and_predict():
    start = time.time()
    train_batch_size = 2048
    val_batch_size = 2048
    test_batch_size = 2048
    max_epoch = 100000
    max_not_improve_time = 100
    device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    model_save_path = 'nn_model.pth'
    train_data_path = 'Train_data.h5'
    test_data_path = 'Test_data.h5'

    # 导入数据
    print('load train data...')
    X_data, Y_data = get_data(train_data_path)
    X_test = torch.from_numpy(get_test_data(test_data_path)).float().to(device)
    test_size = X_test.shape[0]

    # 预测相关
    oof = np.zeros(X_data.shape[0])
    predictions = np.zeros(test_size)

    flod_min_errors = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
        # 数据相关
        print('fold {}'.format(fold_))
        x_train, x_val, y_train, y_val = X_data[trn_idx], X_data[val_idx], Y_data[trn_idx], Y_data[val_idx]
        train_size, val_size = x_train.shape[0], x_val.shape[0]
        x_train, x_val, y_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(x_val).float().to(device), torch.from_numpy(y_train).float().to(device)

        train_batch_time = train_size // train_batch_size
        val_batch_time = val_size // val_batch_size
        test_batch_time = test_size // test_batch_size

        # 模型相关
        model = UsedCarPred(54).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_step = make_train_step(model, criterion, optimizer)

        best_model = None
        errors = []
        losses = []
        not_improve_time = 0
        min_error = 100000.0

        for epoch in range(max_epoch):
            # 训练
            loss = 0.0
            for i in range(train_batch_time+1):
                input = x_train[i*train_batch_size: min((i+1)*train_batch_size, train_size)]
                pred = y_train[i*train_batch_size: min((i+1)*train_batch_size, train_size)]
                loss += train_step(input, pred)
            losses.append(loss)

            # 验证
            model.eval()
            val_pred = np.zeros(val_size)
            for i in range(val_batch_time+1):
                input = x_val[i*val_batch_size: min((i+1)*val_batch_size, val_size)]
                val_pred[i * val_batch_size: min((i+1)*val_batch_size, val_size)] = model(input).detach().cpu().numpy()
            error = mean_absolute_error(val_pred, y_val)
            errors.append(error)

            # 长时间未提高则终止训练
            if error >= min_error:
                not_improve_time += 1
                if not_improve_time >= max_not_improve_time:
                    print('Not improve {} times, stop training.'.format(max_not_improve_time))
                    print('Best iteration error: {}'.format(min_error))
                    flod_min_errors.append(min_error)
                    break
            else:
                not_improve_time = 0
                min_error = error
                best_model = model
                # 保存模型
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                # }, model_save_path)

            if epoch % 50 == 0:
                print('epoch: {} - loss: {} - val_error: {}'.format(epoch, loss, error))

        # 验证集预测
        # model = UsedCarPred(54)
        # checkpoint = torch.load(model_save_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.to(device)

        best_model.eval()
        val_pred = np.zeros(val_size)
        for i in range(val_batch_time+1):
            input = x_val[i*val_batch_size: min((i+1)*val_batch_size, val_size)]
            val_pred[i*val_batch_size: min((i+1)*val_batch_size, val_size)] = best_model(input).detach().cpu().numpy()
        oof[val_idx] = val_pred

        # 测试集预测
        best_model.eval()
        test_pred = np.zeros(test_size)
        for i in range(test_batch_time+1):
            input = X_test[i*test_batch_size: min((i+1)*test_batch_size, test_size)]
            test_pred[i*test_batch_size: min((i+1)*test_batch_size, test_size)] = best_model(input).detach().cpu().numpy()
        predictions += test_pred / folds.n_splits

        # plt.plot([i for i in range(len(errors))], errors)
        # plt.xlabel('epoch')
        # plt.ylabel('error')
        # plt.show()

    print(flod_min_errors)
    total_error = mean_absolute_error(Y_data, oof)
    print(total_error)
    print('time: {}'.format(time.time()-start))

    result = pd.DataFrame()
    result['SaleID'] = np.arange(150000, 200000, 1)
    result['price'] = predictions
    save_name = 'nn_pred.csv'
    save_path = 'result/' + save_name
    result.to_csv(save_path, index=False)
    print('保存成功')






def predict():
    model_load_path = 'nn_model.pth'
    test_data_path = 'Test_data.h5'

    model = UsedCarPred(54)
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('load test data...')
    X_test = get_test_data(test_data_path)

    prediction = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        prediction[i] = model(X_test[i]).item()

    result = pd.DataFrame()
    result['SaleID'] = np.arange(150000, 200000, 1)
    result['price'] = prediction
    save_name = input('请输入保存的文件的名字(.csv)')
    save_path = 'result/' + save_name
    result.to_csv(save_path, index=False)
    print('保存成功')


if __name__ == '__main__':
    train_and_predict()

















