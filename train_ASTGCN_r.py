#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/METR_LA_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)


train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)
            
        # Ensure val_loss is a scalar value
        if isinstance(val_loss, tuple):
            val_loss = val_loss[0]  # Assuming the first element is the main loss value

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)


            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor,metric_method ,_mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type)


import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import torch

# Funzione per salvare e visualizzare i grafici
def save_and_show_plot(fig, filename):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.show()
  
def plot_errors(predictions, true_values):
    """
    Plotta i risultati delle previsioni rispetto ai valori reali, evidenziando i casi con errori massimi e minimi.
    
    :param predictions: array con le previsioni del modello
    :param true_values: array con i valori reali
    """
    # Verifica delle dimensioni
    print(f'Dimensione delle previsioni: {predictions.shape}')
    print(f'Dimensione dei valori reali: {true_values.shape}')

    # Calcolo dell'errore assoluto
    errors = np.abs(predictions - true_values)

    # Appiattiamo l'array degli errori per trovare il massimo e il minimo in tutte le dimensioni
    flattened_errors = errors.flatten()

    # Trova i punti con errore maggiore e minore
    max_error_idx = np.argmax(flattened_errors)
    min_error_idx = np.argmin(flattened_errors)

    print(f'Punto con errore massimo: {max_error_idx}, Errore: {flattened_errors[max_error_idx]}')
    print(f'Punto con errore minimo: {min_error_idx}, Errore: {flattened_errors[min_error_idx]}')

    # Riformattiamo l'indice per ottenere la posizione corretta nelle dimensioni originali
    max_error_idx_unravel = np.unravel_index(max_error_idx, errors.shape)
    min_error_idx_unravel = np.unravel_index(min_error_idx, errors.shape)

    # Plot delle previsioni vs valori reali
    plt.figure(figsize=(14, 7))
    plt.plot(true_values.flatten(), label="Valori Reali", color='blue', alpha=0.6)
    plt.plot(predictions.flatten(), label="Previsioni", color='orange', alpha=0.6)

    # Evidenzia i punti con errore massimo e minimo
    plt.scatter(max_error_idx_unravel[0], predictions[max_error_idx_unravel], color='red', label="Errore massimo", s=100)
    plt.scatter(min_error_idx_unravel[0], predictions[min_error_idx_unravel], color='green', label="Errore minimo", s=100)

    plt.title("Confronto Previsioni vs Valori Reali")
    plt.xlabel("Punto di osservazione")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)
    plt.show()




# Funzione migliorata per plottare con zoom sugli errori significativi
def plot_sample_output(outputs, labels):
    sample_output = outputs[0]  # Prendiamo il primo campione per semplicità
    sample_label = labels[0]
    
    # Creare un array di errori assoluti per ogni punto
    errors = np.abs(sample_output - sample_label)
    
    # Trova il punto con errore massimo e minimo
    max_error_idx = np.argmax(errors)
    min_error_idx = np.argmin(errors)
    
    # Crea una figura per l'intero dataset
    fig1, ax1 = plt.subplots()
    ax1.plot(sample_label, color='blue', label='Valori Reali')
    ax1.plot(sample_output, color='orange', label='Previsioni')
    ax1.scatter(max_error_idx, sample_output[max_error_idx], color='red', label='Errore massimo')
    ax1.scatter(min_error_idx, sample_output[min_error_idx], color='green', label='Errore minimo')
    ax1.set_title('Confronto Previsioni vs Valori Reali')
    ax1.set_xlabel('Punto di osservazione')
    ax1.set_ylabel('Valore')
    ax1.legend()
    
    # Salva e visualizza il grafico dell'intero dataset
    save_and_show_plot(fig1, 'confronto_completo.png')

    # Zoom su una porzione del dataset per una visualizzazione più chiara
    zoom_range = 500  # Definiamo il numero di punti da visualizzare in dettaglio
    fig2, ax2 = plt.subplots()
    ax2.plot(range(zoom_range), sample_label[:zoom_range], color='blue', label='Valori Reali')
    ax2.plot(range(zoom_range), sample_output[:zoom_range], color='orange', label='Previsioni')
    
    # Troviamo gli errori massimi e minimi nella porzione selezionata
    max_error_idx_zoom = np.argmax(errors[:zoom_range])
    min_error_idx_zoom = np.argmin(errors[:zoom_range])
    
    ax2.scatter(max_error_idx_zoom, sample_output[max_error_idx_zoom], color='red', label='Errore massimo')
    ax2.scatter(min_error_idx_zoom, sample_output[min_error_idx_zoom], color='green', label='Errore minimo')
    ax2.set_title(f'Zoom sui primi {zoom_range} punti')
    ax2.set_xlabel('Punto di osservazione')
    ax2.set_ylabel('Valore')
    ax2.legend()
    
    # Salva e visualizza il grafico con zoom
    save_and_show_plot(fig2, f'confronto_zoom_{zoom_range}.png')

# Esempio di chiamata a questa funzione dopo aver ottenuto previsioni e valori reali
plot_sample_output(predictions, true_values)


def predict_and_evaluate(net, data_loader, data_target_tensor, metric_method, _mean, _std, params_path=None, global_step=0):
    """
    Esegue le previsioni, visualizza gli errori e plotta i campioni di output.
    
    :param net: modello addestrato
    :param data_loader: dataloader dei dati di test
    :param data_target_tensor: tensore con i valori reali
    :param metric_method: funzione per il calcolo della metrica
    :param _mean: media dei dati
    :param _std: deviazione standard dei dati
    :param params_path: percorso per salvare i risultati (deve essere una stringa valida)
    :param global_step: step globale di training o predefinito
    """
    # Imposta un percorso predefinito se params_path è None
    if params_path is None:
        params_path = "./output"
    
    # Controlla e crea la directory se non esiste
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    # Eseguiamo le previsioni e passiamo il global_step
    predictions, true_values = predict_and_save_results_mstgcn(
    net, test_loader, test_target_tensor, global_step, metric_method, _mean, _std, params_path, "test"
    )


    # Converto da tensore a numpy array se necessario
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()

    # Plot degli errori
    plot_errors(predictions, true_values)
    
    # Plot del campione di previsioni rispetto ai valori reali
    plot_sample_output(outputs=predictions, labels=true_values)

# Modifica la chiamata nella funzione principale
if __name__ == "__main__":
    # Caricamento dati
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size
    )

    # Definizione del modello
    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices)

    # Addestramento del modello
    train_main()

    # Previsioni e valutazioni
    predict_and_evaluate(net, test_loader, test_target_tensor, metric_method, _mean, _std)




    # predict_main(13, test_loader, test_target_tensor,metric_method, _mean, _std, 'test')














