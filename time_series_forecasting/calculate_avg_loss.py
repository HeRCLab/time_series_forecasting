from statistics import mean

filename_suffix = 'training_log_vH12_sr001_SW_F_Ep2000_BS256_lr2'
filepath = '/share/iahmad/PycharmProjects/time_series_forecasting/' + filename_suffix + '.txt'


def write_to_file(data, filename):
    f = open(filename + ".txt", "w")
    # data_to_write = data.tolist()
    f.write(str(data).replace('[', '').replace(']', ''))
    f.close()


with open(filepath, 'r') as file:
    data = file.readlines()
    # data = list(map(float, data))
    train_loss_current_epoch = []
    train_loss_all_epochs_dict = {}

    current_epoch = 0
    epoch_data = 0

    for line in data:
        line = line.strip()
        if line.startswith('Epoch '):
            epoch_data = line.split(':')[0].split()[1]
            epoch_data = int(epoch_data)
            # if current_epoch == epoch_data:
        elif line.startswith('#####train_loss') and 'tensor(' in line and 'torch.Size(' not in line:
            # print(line)
            current_loss_val = line.split(', ')[0]
            current_loss_val = current_loss_val.split('(')[1]
            current_loss_val = float(current_loss_val)
            if epoch_data in train_loss_all_epochs_dict.keys():
                train_loss_all_epochs_dict[epoch_data].append(current_loss_val)
            else:
                train_loss_all_epochs_dict[epoch_data] = [current_loss_val]

        # if line.startswith('#####train_loss') or line.startswith('Epoch '):
        #     line = line.strip()
        #     if line.startswith('Epoch '):
        #         epoch_data = line.split(':')[0].split()[1]
        #         epoch_data = int(epoch_data)
        #
        #     elif line.startswith('#####train_loss') and current_epoch == epoch_data:
        #         current_loss_val = line.split(', ')[0]
        #         current_loss_val = current_loss_val.split('(')[1]
        #         train_loss_current_epoch.append(current_loss_val)
        #     elif line.startswith('#####train_loss') and not current_epoch == epoch_data:
        #         train_loss_all_epochs_dict[current_epoch] = train_loss_current_epoch
        #
        #         current_epoch = epoch_data
        #         train_loss_current_epoch = []
        #
        #         current_loss_val = line.split(', ')[0]
        #         current_loss_val = current_loss_val.split('(')[1]
        #         train_loss_current_epoch.append(current_loss_val)

    all_epochs = []
    all_losses = []
    # print(train_loss_all_epochs_dict)
    for epoch in train_loss_all_epochs_dict.keys():
        current_epoch_loss = round(mean(train_loss_all_epochs_dict[epoch]), 4)
        all_epochs.append(epoch)
        all_losses.append(current_epoch_loss)

    print('len(all_epochs) = ', len(all_epochs))
    # print(all_losses)
    write_to_file(all_epochs, 'all_epochs_' + filename_suffix)
    write_to_file(all_losses, 'all_losses_' + filename_suffix)
