import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import json
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pickle import dump
import pickle

from time_series_forecasting.model import TimeSeriesForcasting
import GPUtil
print('GPUtil.showUtilization() = ', GPUtil.showUtilization())
from datetime import datetime
start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('start_time = ', start_time)
torch.cuda.empty_cache()


# def split_df(
#     df: pd.DataFrame, split: str, history_size: int = 120, horizon_size: int = 30
# ):

    # Create a training / validation samples
    # Validation samples are the last horizon_size rows
    #
    # :param df:
    # :param split:
    # :param history_size:
    # :param horizon_size:
    # :return:

    # if split == "train":
    #     end_index = random.randint(horizon_size + 1, df.shape[0] - horizon_size)
    # elif split in ["val", "test"]:
    #     end_index = df.shape[0]
    # else:
    #     raise ValueError

    # label_index = end_index - horizon_size
    # start_index = max(0, label_index - history_size)
    #
    # history = df[start_index:label_index]
    # targets = df[label_index:end_index]
    #
    # return history, targets


# def pad_arr(arr: np.ndarray, expected_size: int = 120):
#     Pad top of array when there is not enough history
#     :param arr:
#     :param expected_size:
#     :return:
#     arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
#     return arr


# def df_to_np(df):
#     arr = np.array(df)
#     arr = pad_arr(arr)
#     return arr

NO_OF_GPUS = 2
LEARNING_RATE = 1e-2
NO_UNIQUE_VALS_IN_DATA = 13020
# NUM_WORKERS = 10
NUM_WORKERS = 40

# BATCH_SIZE = 32
BATCH_SIZE = 128 #512 #128 max for SW T # 256

# MAX_PREV_SAMPLE_SIZE = 16
MAX_PREV_SAMPLE_SIZE = 16

EPOCHS = 75 #2000

IF_PREPARE_DATASET = False
IF_USE_SLIDING_WINDOW_DATASET = True

train_src_data = ''
train_trg_in_data = ''
train_trg_out_data = ''

val_src_data = ''
val_trg_in_data = ''
val_trg_out_data = ''


if IF_USE_SLIDING_WINDOW_DATASET:
    dropbear_Xtrain_filename = 'train_src_sliding_window_True.pkl'
    dropbear_ytrain_in_filename = 'train_trg_in_sliding_window_True.pkl'
    dropbear_ytrain_out_filename = 'train_trg_out_sliding_window_True.pkl'

    dropbear_Xval_filename = 'val_src_sliding_window_True.pkl'
    dropbear_yval_in_filename = 'val_trg_in_sliding_window_True.pkl'
    dropbear_yval_out_filename = 'val_trg_out_sliding_window_True.pkl'

    dropbear_Xtest_filename = ''
    dropbear_ytest_in_filename = ''
    dropbear_ytest_out_filename = ''

    dropbear_all_data_Xtrain_filename = 'all_data_final_test_src_sliding_window_True.pkl'
    dropbear_all_data_ytrain_in_filename = 'all_data_final_test_trg_in_sliding_window_True.pkl'
    dropbear_all_data_ytrain_out_filename = 'all_data_final_test_trg_out_sliding_window_True.pkl'

else:
    dropbear_Xtrain_filename = 'train_src_sliding_window_False.pkl'
    dropbear_ytrain_in_filename = 'train_trg_in_sliding_window_False.pkl'
    dropbear_ytrain_out_filename = 'train_trg_out_sliding_window_False.pkl'

    dropbear_Xval_filename = 'val_src_sliding_window_False.pkl'
    dropbear_yval_in_filename = 'val_trg_in_sliding_window_False.pkl'
    dropbear_yval_out_filename = 'val_trg_out_sliding_window_False.pkl'

    dropbear_Xtest_filename = ''
    dropbear_ytest_in_filename = ''
    dropbear_ytest_out_filename = ''

    dropbear_all_data_Xtrain_filename = 'all_data_final_test_src_sliding_window_False.pkl'
    dropbear_all_data_ytrain_in_filename = 'all_data_final_test_trg_in_sliding_window_False.pkl'
    dropbear_all_data_ytrain_out_filename = 'all_data_final_test_trg_out_sliding_window_False.pkl'


def write_to_file(data, filename):
    # f = open(filename+".txt", "w")
    # # data_to_write = data.tolist()
    # f.write(str(data))
    # f.close()

    # with open(filename+'.pkl', 'wb') as file:
    file = open(filename+'.pkl', 'wb')
    dump(data, file)
    file.close()


def read_file(filename):
    # with open(filename, 'r') as file:
    #     data = file.read().split(', ')
    #     data = list(map(float, data))
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def prepare_dataset(source_data_list, target_data_list, feature_len, filename_tag):
    src = []
    trg_in = []
    trg_out = []

    if IF_USE_SLIDING_WINDOW_DATASET:
        # if self.feature_len < len(self.target_data_list) + 1:


        for index in range(0, len(source_data_list) - feature_len):
            # print('i = ', i)
            # print('index = ', index)
            src_segment = source_data_list[index:index + feature_len]
            src.append(src_segment)
            # print('tmplist = ', tmplist)

        # src DONE

        for index in range(0, len(target_data_list) - feature_len):
            # print('i = ', i)
            # print('index = ', index)
            trg_in_segment = target_data_list[index:index + feature_len]
            trg_in.append(trg_in_segment)
            # print('tmplist = ', tmplist)

            # trg_in DONE

        # for i in target_data_list[feature_len:]:
        #     trg_out.append(i)
        trg_out = target_data_list[feature_len:]  # DONE





    else:
        list_len = len(source_data_list)
        for indx, stepindx in enumerate(range(0, len(source_data_list), feature_len)):
            # print('stepindx = ', stepindx)
            # print('indx = ', indx)
            # tmplist = mylist[i:i+bptt]
            # print('tmplist = ', tmplist)

            if stepindx + feature_len >= list_len:
                break

            src_segment = source_data_list[stepindx:stepindx + feature_len]
            src.append(src_segment)
            # src DONE

            trg_in_segment = target_data_list[stepindx:stepindx + feature_len]
            trg_in.append(trg_in_segment)
            # trg_in DONE

            trg_out.append(target_data_list[stepindx + feature_len])  # DONE

    write_to_file(src, filename_tag + '_src_sliding_window_' + str(IF_USE_SLIDING_WINDOW_DATASET))
    write_to_file(trg_in, filename_tag + '_trg_in_sliding_window_' + str(IF_USE_SLIDING_WINDOW_DATASET))
    write_to_file(trg_out, filename_tag + '_trg_out_sliding_window_' + str(IF_USE_SLIDING_WINDOW_DATASET))

    return #src, trg_in, trg_out

def split_train_test_val_dataset(X_train, y_train, test_data_percentage = 0.2, validation_data_percentage = 0.2):
    X_train_length = len(X_train)
    y_train_length = len(y_train)

    X_test_length = int(X_train_length * test_data_percentage)
    y_test_length = int(y_train_length * test_data_percentage)
    X_train_length = X_train_length - X_test_length
    y_train_length = y_train_length - y_test_length

    X_val_length = int(X_train_length * validation_data_percentage)
    y_val_length = int(y_train_length * validation_data_percentage)
    X_train_length = X_train_length - X_val_length
    y_train_length = y_train_length - y_val_length

    X_train_full = X_train
    y_train_full = y_train

    X_train = X_train_full[:X_train_length]
    y_train = y_train_full[:y_train_length]

    # To compare with Nile's results
    # X_train = X_train_full[:]
    # y_train = y_train_full[:]

    X_val = X_train_full[X_train_length:X_train_length + X_val_length]
    y_val = y_train_full[y_train_length:y_train_length + y_val_length]

    X_test = X_train_full[X_train_length + X_val_length:]
    y_test = y_train_full[y_train_length + y_val_length:]

    return X_train, y_train, X_val, y_val, X_test, y_test




# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, groups, grp_by, split, features, target):
#         self.groups = groups
#         self.grp_by = grp_by
#         self.split = split
#         self.features = features
#         self.target = target
#
#     def __len__(self):
#         return len(self.groups)
#
#     def __getitem__(self, idx):
#         group = self.groups[idx]
#
#         df = self.grp_by.get_group(group)
#
#         src, trg = split_df(df, split=self.split)
#
#         src = src[self.features + [self.target]]
#
#         src = df_to_np(src)
#
#         trg_in = trg[self.features + [f"{self.target}_lag_1"]]
#
#         trg_in = np.array(trg_in)
#         trg_out = np.array(trg[self.target])
#
#         src = torch.tensor(src, dtype=torch.float)
#         trg_in = torch.tensor(trg_in, dtype=torch.float)
#         trg_out = torch.tensor(trg_out, dtype=torch.float)
#
#         return src, trg_in, trg_out

# ifty
class Dataset(torch.utils.data.Dataset):
    # def __init__(self, groups, grp_by, split, features, target):
    #     self.groups = groups
    #     self.grp_by = grp_by
    #     self.split = split
    #     self.features = features
    #     self.target = target

    def __init__(self, source_data_list, target_in_data_list, target_out_data_list):
        self.source_data_list = source_data_list
        self.target_in_data_list = target_in_data_list
        self.target_out_data_list = target_out_data_list



    def __len__(self):
        # src = read_file(self.source_data_list)

        return len(self.source_data_list)

    def __getitem__(self, idx):
        # print('def __getitem__ called')
        src = self.source_data_list
        trg_in = self.target_in_data_list
        trg_out = self.target_out_data_list
        # group = self.groups[idx]
        #
        # df = self.grp_by.get_group(group)
        #
        # src, trg = split_df(df, split=self.split)
        #
        # src = src[self.features + [self.target]]
        #
        # src = df_to_np(src)
        #
        # trg_in = trg[self.features + [f"{self.target}_lag_1"]]


        # if IF_USE_SLIDING_WINDOW_DATASET:
        #     # if self.feature_len < len(self.target_data_list) + 1:
        #
        #     for index in range(0, len(self.source_data_list_name) - self.feature_len):
        #         # print('i = ', i)
        #         # print('index = ', index)
        #         src_segment = self.source_data_list_name[index:index + self.feature_len]
        #         src.append(src_segment)
        #         # print('tmplist = ', tmplist)
        #
        #     # src DONE
        #
        #
        #     for index in range(0, len(self.target_in_data_list_name) - self.feature_len):
        #         # print('i = ', i)
        #         # print('index = ', index)
        #         trg_in_segment = self.target_in_data_list_name[index:index + self.feature_len]
        #         trg_in.append(trg_in_segment)
        #             # print('tmplist = ', tmplist)
        #
        #         # trg_in DONE
        #
        #
        #     trg_out = self.target_in_data_list_name[self.feature_len:] #DONE
        #
        # else:
        #     list_len = len(self.source_data_list_name)
        #     for indx, stepindx in enumerate(range(0, len(self.source_data_list_name), self.feature_len)):
        #         # print('stepindx = ', stepindx)
        #         # print('indx = ', indx)
        #         # tmplist = mylist[i:i+bptt]
        #         # print('tmplist = ', tmplist)
        #
        #         if stepindx + self.feature_len >= list_len:
        #             break
        #
        #         src_segment = self.source_data_list_name[stepindx:stepindx + self.feature_len]
        #         src.append(src_segment)
        #         # src DONE
        #
        #         trg_in_segment = self.target_in_data_list_name[stepindx:stepindx + self.feature_len]
        #         trg_in.append(trg_in_segment)
        #         # trg_in DONE
        #
        #         trg_out.append(self.target_in_data_list_name[stepindx + self.feature_len]) # DONE


            # trg_out = self.target_data_list[self.feature_len:]


        # else:
        #     # if not enough data # need to check if may cause any issue #Ifty
        #     src = []
        #     trg_in = []
        #     trg_out = []



        src = np.array(src)
        trg_in = np.array(trg_in)
        trg_out = np.array(trg_out)


        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        # # srctmp = torch.tensor(src)
        # print("torch.max(srctmp[0]):", torch.max(src[0]))
        # # trgtmp = torch.tensor(trg_in)
        # print("torch.max(trgtmp[0]):", torch.max(trg_in[0]))
        # # trg_outtmp = torch.tensor(trg_out)
        # print("torch.max(trg_outtmp[0]):", torch.max(trg_out[0]))

        # print('2. @Dataset @__getitem__ len(src) = ', len(src))
        # print('2. @Dataset @__getitem__ len(trg_in) = ', len(trg_in))
        # print('2. @Dataset @__getitem__ len(trg_out) = ', len(trg_out))

        # print('2. @Dataset @__getitem__ src = ', np.array(src))
        # print('2. @Dataset @__getitem__ trg_in = ', np.array(trg_in))
        # print('2. @Dataset @__getitem__ trg_out = ', np.array(trg_out))

        return src, trg_in, trg_out

# data_csv_path = x_train_data_path
# feature_target_names_path = y_train_data_path
def train(
    x_train_data_path: str,
    y_train_data_path: str,


    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    # batch_size: int = 32,
    batch_size: int = BATCH_SIZE, #128,

        # epochs: int = 2000,
    epochs: int = EPOCHS, #100,
    horizon_size: int = 30,
):
    # data = pd.read_csv(data_csv_path)

    data_dict_Xtrain_ytrain = {}


    # ifty
    with open(x_train_data_path, 'r') as file:
        dropbear_Xtrain = file.read().split(', ')
        dropbear_Xtrain = list(map(float, dropbear_Xtrain))
        file.close()

    with open(y_train_data_path, 'r') as file:
        dropbear_ytrain = file.read().split(', ')
        dropbear_ytrain = list(map(float, dropbear_ytrain))
        file.close()


    # dropbear_Xtrain_ytrain = dropbear_Xtrain + dropbear_ytrain
    # dropbear_Xtrain_ytrain_set = set(dropbear_Xtrain_ytrain)
    # start_tensor_val = 1
    # for item in dropbear_Xtrain_ytrain_set:
    #     # data_dict_Xtrain_ytrain[dropbear_Xtrain_ytrain_set[i]] = i
    #     data_dict_Xtrain_ytrain[item] = start_tensor_val
    #     start_tensor_val = start_tensor_val + 1
    #
    #
    # print('data_dict_Xtrain_ytrain = ', data_dict_Xtrain_ytrain)
    #
    # print('before dropbear_Xtrain = ', dropbear_Xtrain)
    # for i in range(len(dropbear_Xtrain)):
    #     float_val = dropbear_Xtrain[i]
    #     dropbear_Xtrain[i] = data_dict_Xtrain_ytrain[float_val]
    #
    # print('after dropbear_Xtrain = ', dropbear_Xtrain)
    #
    # print('before dropbear_ytrain = ', dropbear_ytrain)
    # for i in range(len(dropbear_ytrain)):
    #     float_val = dropbear_ytrain[i]
    #     dropbear_ytrain[i] = data_dict_Xtrain_ytrain[float_val]
    #
    # print('after dropbear_ytrain = ', dropbear_ytrain)



    X_traintmp = np.asarray(dropbear_Xtrain, dtype=float)

    # srctmp = torch.tensor(X_traintmp, dtype=torch.float)
    # print("torch.max(srctmp[0]):", torch.max(srctmp[0]))

    y_traintmp = np.asarray(dropbear_ytrain, dtype=float)

    # trgtmp = torch.tensor(y_traintmp, dtype=torch.float)
    # print("torch.max(trgtmp[0]):", torch.max(trgtmp[0]))

    # merge X, y to get combined vocab
    X_y_traintmp = np.concatenate([X_traintmp, y_traintmp])
    no_unique_vals_in_data = len(set(X_y_traintmp))
    print('vvi. no_unique_vals_in_data = ', no_unique_vals_in_data)
    # print()


    # NEED to call only once #WHENEVER DATASET CHANGES
    if IF_PREPARE_DATASET:
        prepare_dataset(dropbear_Xtrain, dropbear_ytrain, MAX_PREV_SAMPLE_SIZE, 'all_data_final_test')

        dropbear_Xtrain, dropbear_ytrain, dropbear_Xval, dropbear_yval, dropbear_Xtest, dropbear_ytest = split_train_test_val_dataset(dropbear_Xtrain, dropbear_ytrain)
        prepare_dataset(dropbear_Xtrain, dropbear_ytrain, MAX_PREV_SAMPLE_SIZE, 'train')
        prepare_dataset(dropbear_Xval, dropbear_yval, MAX_PREV_SAMPLE_SIZE, 'val')
        prepare_dataset(dropbear_Xtest, dropbear_ytest, MAX_PREV_SAMPLE_SIZE, 'test')

        return



    # TEST Code # OK
    # src = read_file('train_src_sliding_window_True.pkl')
    # trg_in = read_file('train_trg_in_sliding_window_True.pkl')
    # trg_out = read_file('train_trg_out_sliding_window_True.pkl')
    # print('test data src = ', src)
    # print('test data trg_in = ', trg_in)
    # print('test data trg_out = ', trg_out)
    # print('type(src) =', type(src))
    # print('type(trg_in) =', type(trg_in))
    # print('type(trg_out) =', type(trg_out))
    # print('len(src) = ', len(src))
    # print('len(trg_in) = ', len(trg_in))
    # print('len(trg_out) = ', len(trg_out))



    # with open(feature_target_names_path) as f:
    #     feature_target_names = json.load(f)

    # data_train = data[~data[feature_target_names["target"]].isna()]
    #
    # grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])
    #
    # groups = list(grp_by_train.groups)

    # full_groups = [
    #     grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon_size
    # ]

    # train_data = Dataset(
    #     groups=full_groups,
    #     grp_by=grp_by_train,
    #     split="train",
    #     features=feature_target_names["features"],
    #     target=feature_target_names["target"],
    # )
    # return

    train_data = Dataset(
        source_data_list=train_src_data,
        target_in_data_list=train_trg_in_data,
        target_out_data_list=train_trg_out_data
    )


    # val_data = Dataset(
    #     groups=full_groups,
    #     grp_by=grp_by_train,
    #     split="val",
    #     features=feature_target_names["features"],
    #     target=feature_target_names["target"],
    # )

    val_data = Dataset(
        source_data_list=val_src_data,
        target_in_data_list=val_trg_in_data,
        target_out_data_list=val_trg_out_data

    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    # print('@train train_data = ', train_data)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False, #may need to make False
    )
    # print('@train train_loader = ', train_loader)
    #
    # print('@train val_data = ', val_data)



    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False, #may need to make False
    )
    # print('@train val_loader = ', val_loader)



    # need to change below
    # model = TimeSeriesForcasting(
    #     n_encoder_inputs=len(feature_target_names["features"]) + 1,
    #     n_decoder_inputs=len(feature_target_names["features"]) + 1,
    #     lr=1e-5,
    #     dropout=0.1,
    # )

    model = TimeSeriesForcasting(
        n_encoder_inputs=MAX_PREV_SAMPLE_SIZE,
        n_decoder_inputs=MAX_PREV_SAMPLE_SIZE,
        no_unique_vals_in_data= NO_UNIQUE_VALS_IN_DATA, #30657, #40,#30657, #no_unique_vals_in_data,
        lr=LEARNING_RATE,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=NO_OF_GPUS,
        accelerator='ddp',
        logger=logger,
        callbacks=[checkpoint_callback],
    )



    trainer.fit(model, train_loader, val_loader)
    # temp 9/12/23
    # trainer.fit(model, train_loader, train_loader)



    result_val = trainer.test(test_dataloaders=val_loader)
    print('@train result_val = ', result_val)
    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)


    # ifty end

    return output_json

# def read_all_prepared_dataset():
#     train_src_data = read_file(dropbear_Xtrain_filename)
#     train_trg_in_data = read_file(dropbear_ytrain_in_filename)
#     train_trg_out_data = read_file(dropbear_ytrain_out_filename)
#
#     val_src_data = read_file(dropbear_Xval_filename)
#     val_trg_in_data = read_file(dropbear_yval_in_filename)
#     val_trg_out_data = read_file(dropbear_yval_out_filename)



if __name__ == "__main__":
    print('check ifty 0')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")

    parser.add_argument("--feature_target_names_path")

    parser.add_argument("--output_json_path", default=None)

    parser.add_argument("--log_dir")

    parser.add_argument("--model_dir")

    # parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    args = parser.parse_args()

    # read all data
    train_src_data = read_file(dropbear_Xtrain_filename)
    train_trg_in_data = read_file(dropbear_ytrain_in_filename)
    train_trg_out_data = read_file(dropbear_ytrain_out_filename)

    val_src_data = read_file(dropbear_Xval_filename)
    val_trg_in_data = read_file(dropbear_yval_in_filename)
    val_trg_out_data = read_file(dropbear_yval_out_filename)


    train(
        x_train_data_path=args.data_csv_path,
        y_train_data_path=args.feature_target_names_path,

        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
    )


# cursor
"""
"""

end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('end_time = ', end_time)
