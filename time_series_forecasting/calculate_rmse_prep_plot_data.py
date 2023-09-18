from pytorch_lightning.utilities.cloud_io import load as pl_load
import pytorch_lightning as pl
from pickle import dump
import pickle
import numpy as np
import torch
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from time_series_forecasting.model import TimeSeriesForcasting
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

IF_USE_SLIDING_WINDOW_DATASET = False
LEARNING_RATE = 1e-3
NO_UNIQUE_VALS_IN_DATA = 13020

# MAX_PREV_SAMPLE_SIZE = 16
MAX_PREV_SAMPLE_SIZE = 16
NUM_WORKERS = 1

# BATCH_SIZE = 32
BATCH_SIZE = 256 #128 max for SW T # 256
# src_data = read_file(dropbear_Xtrain_filename)
# trg_in_data = read_file(dropbear_ytrain_in_filename)
# trg_out_data = read_file(dropbear_ytrain_out_filename)

filepath = '/share/iahmad/PycharmProjects/time_series_forecasting/'

if IF_USE_SLIDING_WINDOW_DATASET:
    dropbear_all_data_Xtrain_filename = filepath+'all_data_final_test_src_sliding_window_True.pkl'
    dropbear_all_data_ytrain_in_filename = filepath+'all_data_final_test_trg_in_sliding_window_True.pkl'
    dropbear_all_data_ytrain_out_filename = filepath+'all_data_final_test_trg_out_sliding_window_True.pkl'

else:
    dropbear_all_data_Xtrain_filename = filepath+'all_data_final_test_src_sliding_window_False.pkl'
    dropbear_all_data_ytrain_in_filename = filepath+'all_data_final_test_trg_in_sliding_window_False.pkl'
    dropbear_all_data_ytrain_out_filename = filepath+'all_data_final_test_trg_out_sliding_window_False.pkl'

trained_config_json_filepath = '/share/iahmad/PycharmProjects/time_series_forecasting/models/trained_config.json'

with open(trained_config_json_filepath) as f:
    trained_config_json = json.load(f)

ckpt_path = trained_config_json['best_model_path']

# checkpoint_callback = ModelCheckpoint(
#     monitor="valid_loss",
#     mode="min",
#     dirpath='',
#     filename="ts",
# )
# logger = TensorBoardLogger(
#     save_dir='',
# )
# trainer = pl.Trainer(
#     max_epochs=100,
#     gpus=4,
#     accelerator='ddp',
#     logger=logger,
#     callbacks=[checkpoint_callback],
# )
def read_file(filename):
    # with open(filename, 'r') as file:
    #     data = file.read().split(', ')
    #     data = list(map(float, data))
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

def arr_dimen(a):
  return [len(a)]+arr_dimen(a[0]) if(type(a) == list) else []

def convert_to_lists_of_float(main_list):
    # assuming 2D i/p list
    final_main_list = []
    list_dim = arr_dimen(main_list)
    print('list_dim = ', list_dim)
    # for i in range(len(list_dim)-1):
    for indx1 in range(list_dim[0]):
        temp_list = main_list[indx1]
        temp_list = list(map(float, temp_list))
        final_main_list.append(temp_list)

    return final_main_list

def prep_data(source_data_list, target_in_data_list, target_out_data_list):
    # print('type(source_data_list) = ', type(source_data_list))
    source_data_list = convert_to_lists_of_float(source_data_list)
    target_in_data_list = convert_to_lists_of_float(target_in_data_list)

    target_out_data_list = list(map(float, target_out_data_list))


    # src = list(map(float, source_data_list))
    # trg_in = list(map(float, target_in_data_list))
    # trg_out = list(map(float, target_out_data_list))
    src = source_data_list
    trg_in = target_in_data_list
    trg_out = target_out_data_list

    final_src = []
    final_trg_in = []
    final_trg_out = []

    mid_dim_src = []
    # for indx, stepindx in enumerate(range(0, len(src), BATCH_SIZE)):
    for indx, stepindx in enumerate(range(0, len(src), BATCH_SIZE)):
        segment = src[stepindx:stepindx+BATCH_SIZE]
        final_src.append(segment)

    for indx, stepindx in enumerate(range(0, len(trg_in), BATCH_SIZE)):
        segment = trg_in[stepindx:stepindx+BATCH_SIZE]
        final_trg_in.append(segment)

    for indx, stepindx in enumerate(range(0, len(trg_out), BATCH_SIZE)):
        segment = trg_out[stepindx:stepindx+BATCH_SIZE]
        final_trg_out.append(segment)



    # print('final_src = ', final_src)
    # print('final_trg_in = ', final_trg_in)
    # print('final_trg_out = ', final_trg_out)

    # final_src = list(map(float, final_src))
    # final_trg_in = list(map(float, final_trg_in))
    # final_trg_out = list(map(float, final_trg_out))

    # src = np.array(final_src, dtype=np.float32)
    # trg_in = np.array(final_trg_in, dtype=np.float32)
    # trg_out = np.array(final_trg_out, dtype=np.float32)

    # src = np.array(final_src)
    # trg_in = np.array(final_trg_in)
    # trg_out = np.array(final_trg_out)
    # list_dim_final_src = arr_dimen(final_src)
    # print('1. list_dim_final_src = ', list_dim_final_src)

    final_src = final_src[:len(final_src)-1]
    final_trg_in = final_trg_in[:len(final_trg_in)-1]
    final_trg_out = final_trg_out[:len(final_trg_out)-1]

    list_dim_final_src = arr_dimen(final_src)
    print('2. list_dim_final_src = ', list_dim_final_src)
    list_dim_final_trg_in = arr_dimen(final_trg_in)
    print('2. list_dim_final_trg_in = ', list_dim_final_trg_in)
    list_dim_final_trg_out = arr_dimen(final_trg_out)
    print('2. list_dim_final_trg_out = ', list_dim_final_trg_out)

    src = torch.tensor(final_src, dtype=torch.float)
    trg_in = torch.tensor(final_trg_in, dtype=torch.float)
    trg_out = torch.tensor(final_trg_out, dtype=torch.float)

    return src, trg_in, trg_out

def write_to_file(data, filename):
    # f = open(filename+".txt", "w")
    # # data_to_write = data.tolist()
    # f.write(str(data))
    # f.close()

    # with open(filename+'.txt', 'wb') as file:
    file = open(filename+'.txt', 'w')
    # dump(data, file)
    file.write(str(data).replace('[','').replace(']',''))
    file.close()
# def test_all_data(model, batch):
#     src, trg_in, trg_out = batch
#
#     y_hat = model((src, trg_in))  # check trg_in
#
#     y_hat = y_hat.view(-1)
#     y = trg_out.view(-1)
#
#     print('@test_all_data y_hat = ', y_hat.detach().cpu().numpy())
#     print('@test_all_data y = ', y.detach().cpu().numpy())
#
#     print('@test_all_data unique vals at y = ', len(set(y.detach().cpu().numpy())))
#     print('@test_all_data unique vals at y_hat = ', len(set(y_hat.detach().cpu().numpy())))
#
#     rms = mean_squared_error(y.detach().numpy(), y_hat.detach().numpy(), squared=False)
#     print('************** rms = ', rms)
#
#     # write_to_file(y_hat, "y_hat_vH11_SW_F_Epoch2000_Batch_Size256lr5")
#     # write_to_file(y, "y_vH11_SW_F_Epoch2000_Batch_Size256lr5")
#
#
#     loss = smape_loss(y_hat, y)
#     print("test_all_data_loss", loss)
#
#     return


src_data = read_file(dropbear_all_data_Xtrain_filename)
trg_in_data = read_file(dropbear_all_data_ytrain_in_filename)
trg_out_data = read_file(dropbear_all_data_ytrain_out_filename)

src, trg_in, trg_out = prep_data(src_data, trg_in_data, trg_out_data)
# print('src = ', src.detach().cpu().numpy())
# print('trg_in = ', trg_in.detach().cpu().numpy())
# print('trg_out = ', trg_out.detach().cpu().numpy())

# print('src.size() = ', src.size())
# print('trg_in_data.size() = ', trg_in_data.size())
# print('trg_out_data.size() = ', trg_out_data.size())


# train_data = prep_data(src_data, trg_in_data, trg_out_data)

# train_loader = DataLoader(
#     train_data,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     shuffle=False,  # may need to make False
# )

# model = TimeSeriesForcasting(
#         n_encoder_inputs=MAX_PREV_SAMPLE_SIZE,
#         n_decoder_inputs=MAX_PREV_SAMPLE_SIZE,
#         no_unique_vals_in_data= NO_UNIQUE_VALS_IN_DATA, #30657, #40,#30657, #no_unique_vals_in_data,
#         lr=LEARNING_RATE,
#         dropout=0.1,
#     )
#
# # model = pl.trainer.lightning_module
# print('ckpt_path = ', ckpt_path)
# ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
#
# model.load_state_dict(ckpt['state_dict'])
# # print('model = ', model)
#
# # print('y = ', read_file(filepath+'y_vH11_SW_F_Epoch200_Batch_Size256lr5.pkl'))
# # print('y_hat = ', read_file(filepath+'y_hat_vH11_SW_F_Epoch200_Batch_Size256lr5.pkl'))
#
# test_all_data(model, (src, trg_in, trg_out))

def prep_plot_datay_data(y_data = read_file(filepath+'y_vH11_SW_F_Epoch2000_Batch_Size256lr5.pkl'), y_hat_data = read_file(filepath+'y_hat_vH11_SW_F_Epoch2000_Batch_Size256lr5.pkl')):
    y_data = y_data.detach().cpu().numpy().tolist()
    y_hat_data = y_hat_data.detach().cpu().numpy().tolist()


    write_to_file(y_data, 'y_vH11_SW_F_Epoch200_Batch_Size256lr5')
    write_to_file(y_hat_data, 'y_hat_vH11_SW_F_Epoch200_Batch_Size256lr5')

prep_plot_datay_data()