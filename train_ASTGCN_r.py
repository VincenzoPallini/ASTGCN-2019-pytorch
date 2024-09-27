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
import torch
import networkx as nx

# Funzione per salvare e visualizzare i grafici
def save_and_show_plot(fig, filename):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.show()

def visualize_node_predictions(predictions, true_values, adj_mx, node_id, k=2, j=3):
    """
    Visualizza le previsioni, i valori reali e gli errori per un nodo specifico e il suo sottografo
    in un intervallo temporale che include lo step precedente.
    
    :param predictions: Array numpy con le previsioni (shape: [batch, num_nodes, num_for_predict])
    :param true_values: Array numpy con i valori reali (shape: [batch, num_nodes, num_for_predict])
    :param adj_mx: Matrice di adiacenza del grafo
    :param node_id: ID del nodo da analizzare
    :param k: Distanza massima per il sottografo (default: 2)
    :param j: Numero di snapshot temporali da visualizzare (default: 3, incluso lo step precedente)
    """
    # Verifica che il nodo esista
    if node_id >= adj_mx.shape[0]:
        raise ValueError(f"Il nodo {node_id} non esiste nella matrice di adiacenza.")
    
    # Creare il grafo da adj_mx
    G = nx.from_numpy_array(adj_mx)
    
    # Ottenere il sottografo centrato sul nodo_id
    subgraph = nx.ego_graph(G, node_id, radius=k)
    subgraph_nodes = list(subgraph.nodes())
    
    # Estrarre i dati per il sottografo
    sub_predictions = predictions[:, subgraph_nodes, :]
    sub_true_values = true_values[:, subgraph_nodes, :]
    
    # Calcolare gli errori
    errors = np.abs(sub_predictions - sub_true_values)
    
    # Controllo che j non superi il numero di snapshot disponibili
    num_snapshots = predictions.shape[2]
    j = min(j, num_snapshots)
    
    # Creare il layout del grafico
    fig, axes = plt.subplots(j, 3, figsize=(15, 5 * j))
    fig.suptitle(f"Analisi del nodo {node_id} e del suo sottografo (distanza {k})")
    
    # Aggiungiamo uno snapshot precedente all'intervallo temporale
    t_pred = num_snapshots - 1  # Ultimo tempo di previsione
    times = list(range(max(0, t_pred - j + 1), t_pred + 1))  # Aggiunge il tempo precedente
    
    for i, t in enumerate(times):
        # Print the values as requested
        print("Valori ground truth:", sub_true_values[0, :, t])
        print("Valori previsione:", sub_predictions[0, :, t])
        print("Errori:", errors[0, :, t])

        # Ground Truth
        nx.draw(subgraph, ax=axes[i, 0], node_color=sub_true_values[0, :, t], cmap='viridis', 
                with_labels=True, node_size=500)
        axes[i, 0].set_title(f"Ground Truth (t={t})")
        
        # Previsione
        nx.draw(subgraph, ax=axes[i, 1], node_color=sub_predictions[0, :, t], cmap='viridis', 
                with_labels=True, node_size=500)
        axes[i, 1].set_title(f"Previsione (t={t})")
      
        some_small_value = np.max(errors)
      
        # Errore with new colormap and limits as requested
        nx.draw(subgraph, ax=axes[i, 2], node_color=errors[0, :, t], cmap='Reds', 
                with_labels=True, node_size=500, vmin=0, vmax=some_small_value)
        axes[i, 2].set_title(f"Errore (t={t})")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Mantieni lo spazio per il titolo principale
    
    # Creare la directory di output se non esiste
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva il grafico
    plt.savefig(os.path.join(output_dir, f"node_{node_id}_analysis.png"))
    plt.close()

    print(f"Grafico salvato come node_{node_id}_analysis.png nella cartella 'output'")



# Funzione per analizzare e visualizzare gli errori
def analyze_errors(predictions, true_values):
    """
    Analisi e visualizzazione degli errori tra previsioni e valori reali
    """
    # Calcolo degli errori assoluti
    errors = np.abs(predictions - true_values)
    
    # 1. Plot della distribuzione degli errori
    fig1, ax1 = plt.subplots()
    ax1.hist(errors.flatten(), bins=50, color='gray', alpha=0.7)
    ax1.set_title('Distribuzione degli Errori Assoluti')
    ax1.set_xlabel('Errore Assoluto')
    ax1.set_ylabel('Frequenza')
    
    # Salva e mostra il grafico
    save_and_show_plot(fig1, 'distribuzione_errori.png')

    # 2. Trova i punti con errore massimo e minimo
    max_error_idx = np.argmax(errors.flatten())
    min_error_idx = np.argmin(errors.flatten())

    print(f'Errore massimo: {errors.flatten()[max_error_idx]:.2f} al punto {max_error_idx}')
    print(f'Errore minimo: {errors.flatten()[min_error_idx]:.2f} al punto {min_error_idx}')

    # Plot delle previsioni vs valori reali con evidenziazione degli errori massimo e minimo
    fig2, ax2 = plt.subplots()
    ax2.plot(true_values.flatten(), label="Valori Reali", color='blue', alpha=0.6)
    ax2.plot(predictions.flatten(), label="Previsioni", color='orange', alpha=0.6)

    # Evidenzia i punti con errore massimo e minimo
    ax2.scatter(max_error_idx, predictions.flatten()[max_error_idx], color='red', label="Errore massimo", s=100)
    ax2.scatter(min_error_idx, predictions.flatten()[min_error_idx], color='green', label="Errore minimo", s=100)

    ax2.set_title("Confronto Previsioni vs Valori Reali con Errori Massimo e Minimo")
    ax2.set_xlabel("Punto di osservazione")
    ax2.set_ylabel("Valore")
    ax2.legend()
    ax2.grid(True)

    # Salva e mostra il grafico
    save_and_show_plot(fig2, 'confronto_previsioni_errori.png')

    # 3. Plot dell'errore per ogni punto di osservazione
    fig3, ax3 = plt.subplots()
    ax3.plot(errors.flatten(), color='red')
    ax3.set_title('Errore Assoluto per Punto di Osservazione')
    ax3.set_xlabel('Punto di Osservazione')
    ax3.set_ylabel('Errore Assoluto')
    ax3.grid(True)

    # Salva e mostra il grafico
    save_and_show_plot(fig3, 'errore_per_osservazione.png')

    # 4. Analisi statistica degli errori
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"Errore medio: {mean_error:.2f}")
    print(f"Deviazione standard degli errori: {std_error:.2f}")
    print(f"Errore massimo: {max_error:.2f}")
    print(f"Errore minimo: {min_error:.2f}")

def plot_sample_output(outputs, labels):
    print(f"Shape of outputs: {outputs.shape}, Shape of labels: {labels.shape}")
    
    if outputs.shape[0] < 1 or labels.shape[0] < 1:
        print("Non ci sono abbastanza dati per creare il grafico.")
        return
    
    sample_output = outputs[0]  # Prendiamo il primo campione di previsioni
    sample_label = labels[0]  # Prendiamo il primo campione di etichette
    
    print(f"Shape of sample_output: {sample_output.shape}, Shape of sample_label: {sample_label.shape}")
    
    if sample_output.shape != sample_label.shape:
        print("Dimensioni del campione non corrispondenti tra output e etichette.")
        return
    
    # Appiattisci i dati per il calcolo degli errori
    flat_output = sample_output.flatten()
    flat_label = sample_label.flatten()
    
    # Calcolo degli errori assoluti per il campione corrente
    errors = np.abs(flat_output - flat_label)
    
    # Trova il punto con errore massimo e minimo nel campione corrente
    max_error_idx = np.argmax(errors)
    min_error_idx = np.argmin(errors)
    
    print(f"Indice errore massimo: {max_error_idx}, Indice errore minimo: {min_error_idx}")
    print(f"Errore massimo: {errors[max_error_idx]}, Errore minimo: {errors[min_error_idx]}")

    # Crea una figura per l'intero campione
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(flat_label, color='blue', label='Valori Reali', alpha=0.7)
    ax1.plot(flat_output, color='orange', label='Previsioni', alpha=0.7)
    
    # Evidenzia il punto con errore massimo e minimo
    ax1.scatter(max_error_idx, flat_output[max_error_idx], color='red', label='Errore massimo')
    ax1.scatter(min_error_idx, flat_output[min_error_idx], color='green', label='Errore minimo')
    ax1.set_title('Confronto Previsioni vs Valori Reali (Campione)')
    ax1.set_xlabel('Punto di osservazione')
    ax1.set_ylabel('Valore')
    ax1.legend()
    
    # Salva e visualizza il grafico dell'intero campione
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confronto_completo_campione.png'))
    plt.close(fig1)

    # Zoom su una porzione del campione per una visualizzazione più chiara
    zoom_range = min(500, len(flat_output))
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(range(zoom_range), flat_label[:zoom_range], color='blue', label='Valori Reali', alpha=0.7)
    ax2.plot(range(zoom_range), flat_output[:zoom_range], color='orange', label='Previsioni', alpha=0.7)
    
    # Trova il punto con errore massimo e minimo nella porzione zoomata
    zoom_errors = errors[:zoom_range]
    max_error_idx_zoom = np.argmax(zoom_errors)
    min_error_idx_zoom = np.argmin(zoom_errors)
    
    ax2.scatter(max_error_idx_zoom, flat_output[max_error_idx_zoom], color='red', label='Errore massimo (zoom)')
    ax2.scatter(min_error_idx_zoom, flat_output[min_error_idx_zoom], color='green', label='Errore minimo (zoom)')
    ax2.set_title(f'Zoom sui primi {zoom_range} punti del campione')
    ax2.set_xlabel('Punto di osservazione')
    ax2.set_ylabel('Valore')
    ax2.legend()
    
    # Salva e visualizza il grafico con zoom
    plt.savefig(os.path.join(output_dir, f'confronto_zoom_{zoom_range}_campione.png'))
    plt.close(fig2)

    # Grafico dettagliato delle prime 50 finestre temporali
    try:
        fig3, ax3 = plt.subplots(figsize=(30, 4), dpi=80)
        
        for i in range(min(50, sample_output.shape[0])):
            ax3.plot(range(i*12, (i+1)*12), sample_output[i], color='red', alpha=0.7)
            ax3.plot(range(i*12, (i+1)*12), sample_label[i], color='blue', alpha=0.7)
        
        ax3.set_title("Previsioni e Valori Reali per le prime 50 finestre temporali")
        ax3.set_xlabel("Intervallo temporale")
        ax3.set_ylabel("Valore")
        
        output_path = os.path.join(output_dir, "dettagli_previsioni_50_finestre.png")
        plt.savefig(output_path)
        print(f"Grafico salvato in: {output_path}")
        
        plt.close(fig3)
    except Exception as e:
        print(f"Si è verificato un errore durante la creazione o il salvataggio del grafico dettagliato: {e}")

    print("Tutti i grafici sono stati salvati nella cartella 'output'.")

    
    # Mostra il grafico dopo averlo salvato
    plt.show()


# Funzione per previsioni e valutazioni migliorata
def predict_and_evaluate(net, data_loader, data_target_tensor, metric_method, _mean, _std, params_path=None, global_step=0):
    if params_path is None:
        params_path = "./output"
    
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    predictions, true_values = predict_and_save_results_mstgcn(
        net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std, params_path, "test"
    )

    # Converto da tensore a numpy array se necessario
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()

    print(f"Shape of predictions: {predictions.shape}, Shape of true_values: {true_values.shape}")

    # Plot degli errori e analisi
    analyze_errors(predictions, true_values)
    
    # Plot del campione di previsioni rispetto ai valori reali
    plot_sample_output(predictions, true_values)

    # Visualizza previsioni per nodi specifici
    for node_id in [0, 10, 20]:  # Esempio: visualizza per i nodi 0, 10 e 20
        visualize_node_predictions(predictions, true_values, adj_mx, node_id, k=2, j=3)

# Chiamata nella funzione principale
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
    predict_and_evaluate(net, test_loader, test_target_tensor, metric_method, _mean, _std, params_path, global_step=0)





    # predict_main(13, test_loader, test_target_tensor,metric_method, _mean, _std, 'test')














