# Functions needed to train each model
from Model import Model_train
from Model import Model_train_both

# Functions needed to test each model
from Model import Model_test
from Model import Model_test_both

# Functions needed to load and normalize the data for each model
from Both_Helpers.loadData import loadData
from Both_Helpers.normalizeData import normalizeData
from Both_Helpers.splitData import splitData

# Optimization
import optuna
from optuna.trial import TrialState

# Graphing
import matplotlib.pyplot as plt

# Others
import torch
import copy
import numpy as np
import json
import os
from MLP import MLP
from copy import deepcopy


# Global variables
DEVICE = torch.device("cpu")
DATA_BUNDLE = None
DATA_BUNDLE_TEST = None
DATA_BUNDLE_CROSS = None
MODEL_TYPE = None
MODEL_TO_TRAIN = None



# Define a model for optuna to optimize
def create_model(trial):
    ### Suggested parameters of the model
    
    # Number of layers
    n_layers = trial.suggest_int("n_layers", 1, 4)
    
    # Number of nodes per layer and dropout rate per layer
    hidden_layer_nodes = []
    dropout_rates = []
    for i in range(0, n_layers):
        hidden_layer_nodes.append(trial.suggest_int(f"n_nodes_{i}", 32, 2048))
        dropout_rates.append(trial.suggest_float(f"dropout_per{i}", 0.1, 0.5))
    
    # Optimizer
    #optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = "Adam"
    
    # Input and output
    outputShape = 10
    inputShape = 34
        
        
    # Return the model
    return MLP(MODEL_TYPE, lr, inputShape, hidden_layer_nodes, 
               outputShape, dropout_rates, optimizer, DEVICE)




# Objective to minimize for one model
def objective(trial):
    # Get the device for this trial
    try:
        device = trial.user_attrs["device"]
    except KeyError:
        device = "cpu"
    device = torch.device(device)

    # Other paramters
    #batchSize = trial.suggest_int("batchSize", 16, 128)
    batchSize = 64
    
    # Create a model
    model = create_model(trial).to(device)
    
    # Number of epochs
    epochs = trial.suggest_int("epochs", 3000, 5000)

    # Copy the data to this trial
    data_bundle = list(deepcopy(DATA_BUNDLE))
    data_bundle_test = list(deepcopy(DATA_BUNDLE_TEST))
    data_bundle_cross = list(deepcopy(DATA_BUNDLE_CROSS))
    for i in range(0, len(data_bundle)):
        data_bundle[i] = data_bundle[i].to(device)
        data_bundle_test[i] = data_bundle_test[i].to(device)
        data_bundle_cross[i] = data_bundle_cross[i].to(device)
    
    # Train the model
    loss_train = Model_train(data_bundle, model, batchSize, epochs)
    
    # Get the test loss for the model
    loss_test = Model_test(model, data_bundle_test)
    
    # Cross loss
    loss_cross = Model_test(model, data_bundle_cross)
    
    # Save the losses
    trial.set_user_attr("Train loss", loss_train)
    trial.set_user_attr("Test loss", loss_test)
    trial.set_user_attr("Cross loss", loss_cross)
    
    # Optimize on test loss
    return loss_test





# Objective to minimize for both models
def objective_both(trial):
    # Get the device for this trial
    try:
        device = trial.user_attrs["device"]
    except KeyError:
        device = "cpu"
    device = torch.device(device)

    # Other paramters
    #batchSize = trial.suggest_int("batchSize", 16, 128)
    batchSize = 64
    
    # Create a model
    model = create_model(trial).to(device)
    
    # Number of epochs
    epochs = trial.suggest_int("epochs", 3000, 5000)

    # Copy the data to this trial
    data_bundle_M1, data_bundle_M2 = deepcopy(DATA_BUNDLE)
    data_bundle_M1, data_bundle_M2 = list(data_bundle_M1), list(data_bundle_M2)
    data_bundle_test_M1, data_bundle_test_M2 = deepcopy(DATA_BUNDLE_TEST)
    data_bundle_test_M1, data_bundle_test_M2 = list(data_bundle_test_M1), list(data_bundle_test_M2)
    data_bundle_cross_M1, data_bundle_cross_M2 = deepcopy(DATA_BUNDLE_CROSS)
    data_bundle_cross_M1, data_bundle_cross_M2 = list(data_bundle_cross_M1), list(data_bundle_cross_M2)
    for i in range(0, len(data_bundle_M1)):
        data_bundle_M1[i] = data_bundle_M1[i].to(device)
        data_bundle_test_M1[i] = data_bundle_test_M1[i].to(device)
        data_bundle_cross_M1[i] = data_bundle_cross_M1[i].to(device)
    for i in range(0, len(data_bundle_M2)):
        data_bundle_M2[i] = data_bundle_M2[i].to(device)
        data_bundle_test_M2[i] = data_bundle_test_M2[i].to(device)
        data_bundle_cross_M2[i] = data_bundle_cross_M2[i].to(device)
    
    # Train the models
    loss_train, loss_train_M1, loss_train_M2 = Model_train_both(data_bundle_M1, data_bundle_M2, model, batchSize, epochs)
    
    # Get the test loss for the model
    loss_test, loss_test_M1, loss_test_M2 = Model_test_both(model, data_bundle_test_M1, data_bundle_test_M2)
    
    # Cross loss
    loss_cross, loss_cross_M1, loss_cross_M2 = Model_test_both(model, data_bundle_cross_M1, data_bundle_cross_M2)
    
    # Save the losses
    trial.set_user_attr("Train loss", loss_train)
    trial.set_user_attr("Train loss M1", loss_train_M1)
    trial.set_user_attr("Train loss M2", loss_train_M2)
    trial.set_user_attr("Test loss", loss_test)
    trial.set_user_attr("Test loss M1", loss_test_M1)
    trial.set_user_attr("Test loss M2", loss_test_M2)
    trial.set_user_attr("Cross loss", loss_cross)
    trial.set_user_attr("Cross loss M1", loss_cross_M1)
    trial.set_user_attr("Cross loss M2", loss_cross_M2)
    
    # Optimize on test loss
    return loss_test


# Class used to optimize using GPUs
class Objective:
    def __init__(self, gpu_queue):
        # Shared queue to manage GPU IDs.
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()

        # Set the device
        trial.set_user_attr("device", f"cuda:{gpu_id}")
        
        # Run the objective
        if MODEL_TO_TRAIN != 3:
            val = objective(trial)
        else:
            val = objective_both(trial)

        # Return GPU ID to the queue.
        self.gpu_queue.put(gpu_id)

        # Return the loss value as the optimization objective
        return val




def optimize():
    # Paramters
    modelType = "normal"       # Model type can be "normal" or "differences" for energy differences
    model1_data_dir = "Data/Model1" # Directory containing model 1 data
    model2_data_dir = "Data/Model2" # Directory containing model 2 data
    test_size = 0.1 # Percent of data to make the test data
    device = "gpu" # Device to put the model on (gpu or cpu)
    model_to_train = 3 # What models to train and test? (1 for M1, 2 for M2,
                       # 3 to optimize both at the same time)
    n_gpus = 8 # Number of gpus to run the study on
    
    # Optimization params
    n_trials = 400 # Number of trials total (over all GPUs)
    timeout = None
    
    # Global declarations
    global MODEL_TYPE
    MODEL_TYPE = modelType
    global DEVICE
    global DATA_BUNDLE
    global DATA_BUNDLE_TEST
    global DATA_BUNDLE_CROSS
    global MODEL_TO_TRAIN
    MODEL_TO_TRAIN = model_to_train
    
    
    # Get the device
    dev = device
    if not torch.cuda.is_available() and device.lower() == "gpu":
        print("Cuda GPU not availble, defaulting to CPU")
        device = torch.device("cpu")
    elif device.lower() == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    DEVICE = device

    
    # Iterate over all 5 delta values
    #deltaVals = [1, 1.25, 1.5, 1.75, 2]
    deltaVals = [1]
    for delta in deltaVals:
        print(f"Delta = {delta}")
        
        ### Data Loading ###
        print("    Loading data...")
        # Load in the data for model 1
        model1_data = loadData(delta, model1_data_dir, device)
        
        # Load in the data for model 2
        model2_data = loadData(delta, model2_data_dir, device)
        print("    Data loaded!")
        
        
        ### Data Normalization ###
        model1_data = normalizeData(modelType, model1_data, True)
        model2_data = normalizeData(modelType, model2_data, False)
        
        
        ### Test/Train split
        model1_train, model1_test, model1_names_train, model1_names_test = splitData(model1_data, test_size)
        model2_train, model2_test, model2_names_train, model2_names_test = splitData(model2_data, test_size)
        
        
        ### Model 1 Training ###
        if model_to_train == 1:
            print(f"Model 1 - delta {delta} ----------------------------------------")
            DATA_BUNDLE = model1_train
            DATA_BUNDLE_TEST = model1_test
            DATA_BUNDLE_CROSS = model2_test
            
            # Create a study for model 1
            study = optuna.create_study(storage="sqlite:///model1_study.db", direction="minimize")
            
            # Running on the GPU
            if dev == "gpu":
                from dask.distributed import Client
                from dask_cuda import LocalCUDACluster
                from multiprocessing import Manager
                from joblib import parallel_backend
                
                # Multi GPU
                cluster = LocalCUDACluster()
                client = Client(cluster)
                
                with Manager() as manager:
                    # Initialize the queue by adding available GPU IDs.
                    gpu_queue = manager.Queue()
                    for i in range(n_gpus):
                        gpu_queue.put(i)
                        
                    # Run the study
                    with parallel_backend("dask", n_jobs=n_gpus):
                        study.optimize(Objective(gpu_queue), n_trials=n_trials, n_jobs=n_gpus, timeout=timeout)
            else:
                study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            # Trial info
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            print("   Losses: ")
            for key, value in trial.user_attrs.items():
                print("    {}: {}".format(key, value))
            print(study.trials_dataframe())
            print("\n\n\n")
        
        
        
        ### Model 2 Training ###
        elif model_to_train == 2:
            print(f"Model 2 - delta {delta} ----------------------------------------")
            DATA_BUNDLE = model2_train
            DATA_BUNDLE_TEST = model2_test
            DATA_BUNDLE_CROSS = model1_test
            
            # Create a study for model 2
            study = optuna.create_study(storage="sqlite:///model2_study.db", direction="minimize")
            
            # Running on the GPU
            if dev == "gpu":
                from dask.distributed import Client
                from dask_cuda import LocalCUDACluster
                from multiprocessing import Manager
                from joblib import parallel_backend
                
                # Multi GPU
                cluster = LocalCUDACluster()
                client = Client(cluster)
                
                with Manager() as manager:
                    # Initialize the queue by adding available GPU IDs.
                    gpu_queue = manager.Queue()
                    for i in range(n_gpus):
                        gpu_queue.put(i)
                        
                    # Run the study
                    with parallel_backend("dask", n_jobs=n_gpus):
                        study.optimize(Objective(gpu_queue), n_trials=n_trials, n_jobs=n_gpus, timeout=timeout)
            else:
                study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            # Trial info
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            print("   Losses: ")
            for key, value in trial.user_attrs.items():
                print("    {}: {}".format(key, value))
            print("\n\n\n")

        
        ### Both models training ###
        elif model_to_train == 3:
            print(f"Model 1 and 2 - delta {delta} ----------------------------------------")
            DATA_BUNDLE = (model1_train, model2_train)
            DATA_BUNDLE_TEST = (model1_test, model2_test)
            DATA_BUNDLE_CROSS = (model2_test, model1_test)
            
            # Create a study for model 2
            study = optuna.create_study(storage="sqlite:///model3_study.db", direction="minimize")
            
            # Running on the GPU
            if dev == "gpu":
                from dask.distributed import Client
                from dask_cuda import LocalCUDACluster
                from multiprocessing import Manager
                from joblib import parallel_backend
                
                # Multi GPU
                cluster = LocalCUDACluster()
                client = Client(cluster)
                
                with Manager() as manager:
                    # Initialize the queue by adding available GPU IDs.
                    gpu_queue = manager.Queue()
                    for i in range(n_gpus):
                        gpu_queue.put(i)
                        
                    # Run the study
                    with parallel_backend("dask", n_jobs=n_gpus):
                        study.optimize(Objective(gpu_queue), n_trials=n_trials, n_jobs=n_gpus, timeout=timeout)
            else:
                study.optimize(objective_both, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            # Trial info
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                
            print("   Losses: ")
            for key, value in trial.user_attrs.items():
                print("    {}: {}".format(key, value))
            print("\n\n\n")
        
    




if __name__=="__main__":
    optimize()