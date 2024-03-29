import numpy as np; import pandas as pd
import scipy as sp; import scipy.stats as st
import torch; import torch.nn as nn
#use numba's just-in-time compiler to speed things up
from numba import njit
from sklearn.preprocessing import StandardScaler; from sklearn.model_selection import train_test_split
import matplotlib as mp; import matplotlib.pyplot as plt; 
#reset matplotlib stle/parameters
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#plt.style.use('seaborn-deep')
mp.rcParams['agg.path.chunksize'] = 10000
mp.rcParams['axes.linewidth'] = 1
font_legend = 15; font_axes=15
# %matplotlib inline
import copy; import sys; import os
from IPython.display import Image, display
from importlib import import_module
FONTSIZE=18
font = {'family': 'serif', 'weight':'normal', 'size':FONTSIZE}
mp.rc('font', **font)
mp.rc('text',usetex=True)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SIR_data = pd.read_csv('../../data/SIR_data.csv.gz')

def cdf(lst, x):

    count = sum(1 for num in lst if num <= x)
    return count / len(lst)


def flatten_SIR_data(df):
    """ and add cdf column"""
    alpha_l = []
    beta_l = []
    Infected_l = []
    li_l = []
    true_CDF_l=[]
    for rowind, row in df.iterrows():
        li = eval(row['li'])
        
        for lambda_val in li:
            true_CDF_l.append(cdf(li, lambda_val))
    
    for rowind, row in df.iterrows():
    #     Infected = eval(row['I'])
    #     Infected_l.append(Infected)
        
        li = eval(row['li'])
        li_l.append(li)

        
        alpha_r = np.full_like(li, row['alpha'])
        alpha_l.append(alpha_r)
        
        beta_r = np.full_like(li, row['beta'])
        beta_l.append(beta_r)
        
        
    
    # for arr in [alpha_l, beta_l, Infected_l, li_l]:
    #     arr = np.array(arr).flatten()
    alpha_l = np.array(alpha_l).flatten()
    beta_l = np.array(beta_l).flatten()
    # Infected_l = np.array(Infected_l).flatten()
    li_l = np.array(li_l).flatten()
    true_CDF_l = np.array(true_CDF_l).flatten()
    
    data_flat = pd.DataFrame({
        'alpha':alpha_l,
        'beta': beta_l,
        # 'I' :Infected_l,
        'li':li_l,
        'true_CDF':true_CDF_l 
    })
    
    return data_flat


SIR_data_flat = flatten_SIR_data(SIR_data)

def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df[target])
    x = np.array(df[source])
    return t, x

def getwholedata_delta_SIR():
    """ Get train test split arrays"""
    
    data = SIR_data_flat
        
    train_data, test_data = train_test_split(data, test_size=0.1)
    #split the train data (0.8 of whole set) again into 0.8*0.8=0.64 of whole set
    

    train_data = train_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    target='true_CDF'
    # target='y'
    # source = ['theta','nu','theta_hat','N','M']

    # source = ['theta', 'nu', 'lambda', 'true_CDF']
    source = ['alpha', 'beta', 'li']

    train_t, train_x = split_t_x(train_data, target=target, source=source)
    test_t,  test_x  = split_t_x(test_data,  target=target, source=source)
    print('train_t shape = ', train_t.shape, '\n')
    print('train_x shape = ', train_x.shape, '\n')
    
    # if valid:
        #if you want to also make a validation data set
    train_data, valid_data = train_test_split(train_data, test_size=0.015)
    valid_data = valid_data.reset_index(drop=True)
    valid_t, valid_x = split_t_x(valid_data, target=target, source=source)

        
    return train_t, train_x, test_t,  test_x, valid_t, valid_x




train_t, train_x, test_t,  test_x, valid_t, valid_x = getwholedata_delta_SIR()

def get_features_training_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return (batch_x, batch_t)

from torch.autograd import Variable

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)
    
    
class SIR_Model(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) ==0:
                #inital layer has to have size of input features as its input layer
                #its output layer can have any size but it must match the size of the input layer of the next linear layer
                #here we choose its output layer as the hidden size (fully connected)
                layer = nn.Linear(nfeatures, hidden_size)
                # torch.nn.init.xavier_uniform_(layer.weight, gain=60)
                torch.nn.init.xavier_normal_(layer.weight, 
                                             # gain=60
                                            )
                layers.append(layer)
                #batch normalization
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.SiLU())
                # layers.append(SinActivation())
                # layers.append(GroupSort(num_groups=1))
            else:
                #if this is not the first layer (we dont have layers)
                layer = nn.Linear(hidden_size, hidden_size)
                # torch.nn.init.xavier_uniform_(layer.weight, gain=60)
                torch.nn.init.xavier_normal_(layer.weight, 
                                             # gain=60
                                            )
                layers.append(layer)
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                layers.append(nn.SiLU())
                # layers.append(nn.Tanh())
                # layers.append(nn.ReLU())
                # layers.append(SinActivation())
                
                #output layer:
        output_layer = nn.Linear(hidden_size, ntargets)
        torch.nn.init.xavier_uniform_(output_layer.weight)
        layers.append(output_layer) 

        # ONLY IF ITS A CLASSIFICATION, ADD SIGMOID
        layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)
    
    
  
  
def save_model(model, PARAMS, pth_string):
    """pth string is the name of the pth file which is a dictionary of dictionaries"""
    models_path = os.path.join(os.getcwd(), '../../models')
    PATH=os.path.join(models_path, pth_string)
    print(f'saving model with the string : {pth_string}\n')
    torch.save({'PARAMS': PARAMS,
                'model_state_dict': model.state_dict()},
                PATH)
    # print(model)
    
    
class SaveModelCheckpoint:
    """Continuous model-checkpointing class. Updates the latest checkpoint of an object based o validation loss each time its called. 
    """
    def __init__(self, best_valid_loss=np.inf):
        """Initiate an instance of the class based on filename and best_valid_loss/

        Args:
            best_valid_loss (float, optional): Best possible validation loss of a checkpoint object. Defaults to np.inf.
        """
        self.best_valid_loss = best_valid_loss

    def __call__(self, model, current_valid_loss, PARAMS, pth_string):
        """When an object of the calss is called, its validation loss gets updated and the model based 
        on the latest validation loss is saved.

        Args:
            model: utils.RegularizedRegressionModel object.
            current_valid_loss (float): current (latest) validation loss of this model during the training process.
            filename_model (str): filename in which the latest model will be saved. Can be a relative or local path. 
        """
        if current_valid_loss < self.best_valid_loss:
            # update the best loss
            self.best_valid_loss = current_valid_loss
            # filename_model='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
            # filename_model = "Trained_IQNx4_%s_TUNED_2lin_with_noise.dict" % target

            # note that n_iterations is the total n_iterations, we dont want to save a million files for each iteration
            save_model(model, PARAMS, pth_string)
            print(
                f"\nCurrent valid loss: {current_valid_loss};  saved better model in models/{pth_string}"
            )
            # save using .pth object which if a dictionary of dicionaries, so that I can have PARAMS saved in the same file


def RMS(v):
    return (torch.mean(v**2)) ** 0.5

def average_quadratic_loss(f, t, x):
    # f and t must be of the same shape

    # inv = torch.where(t !=0, 1/torch.abs(t), 1)
    
    # inv_RMS = torch.where(t !=0, 1/RMS(t), 1)
    
    return  torch.mean(  (f - t)**2)

# Huber loss function
def huber_loss(f, t, x):
    delta=torch.Tensor([1.0])
    huber_mse = 0.5*(f-t)**2
    huber_mae = delta * (torch.abs(t - f) - 0.5 * delta)
    return torch.where(torch.abs(t - f) <= delta, huber_mse, huber_mae)
    
def Huber_loss(f, t, x):
    return torch.nn.functional.huber_loss(f,t, delta=0.7)

def absolute_error(f,t,x):
    return torch.mean(torch.abs(f-t) )

def kl_divergence_loss(q, p, x):
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(torch.log(p), q)
    return loss


def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float().to(device)
        t = torch.from_numpy(targets).float().to(device)
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t, x)
def train_SIR(model, optimizer, avloss,
          batch_size, 
          n_iterations, traces, 
          step, window, PARAMS, pth_string):
    
    # to keep track of average losses
    xx, yy_t, yy_v, yy_v_avg = traces
    
    model = model.to(device)
    

    train_t, train_x, test_t,  test_x, _, _ = getwholedata_delta_SIR()

    model_checkpoint = SaveModelCheckpoint()
    n = len(test_x)
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" % \
          ('iteration', 'train-set', 'valid-set'))
    
    # training_set_features, training_set_targets, evaluation_set_features, evaluation_set_targets = get_data_sets(simulate_data=False, batchsize=batch_size)
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.

        
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        
        #Harrison-like Loader
        batch_x, batch_t = get_features_training_batch(train_x, train_t, batch_size)
        
        #Or Ali's Loader
        # batch_x, batch_t = next(training_set_features()), next(training_set_targets())
        # batch_x_eval, batch_t_eval = next(evaluation_set_features()), next(evaluation_set_targets())

        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float().to(device)
            t = torch.from_numpy(batch_t).float().to(device)    


        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            
            #using Harrison-like loader
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, test_x[:n], test_t[:n])
            
            model_checkpoint(model=model, current_valid_loss=acc_v, PARAMS=PARAMS, pth_string=pth_string)
            

            yy_t.append(acc_t)
            yy_v.append(acc_v)
            
            # compute running average for validation data
            len_yy_v = len(yy_v)
            if   len_yy_v < window:
                yy_v_avg.append( yy_v[-1] )
            elif len_yy_v == window:
                yy_v_avg.append( sum(yy_v) / window )
            else:
                acc_v_avg  = yy_v_avg[-1] * window
                acc_v_avg += yy_v[-1] - yy_v[-window-1]
                yy_v_avg.append(acc_v_avg / window)
                        
            if len(xx) < 1:
                xx.append(0)
                print("%10d\t%10.6f\t%10.6f" % \
                      (xx[-1], yy_t[-1], yy_v[-1]))
            else:
                xx.append(xx[-1] + step)
                    
                print("\r%10d\t%10.6f\t%10.6f\t%10.6f" % \
                          (xx[-1], yy_t[-1], yy_v[-1], yy_v_avg[-1]), 
                      end='')
            
    print()      
    return (xx, yy_t, yy_v, yy_v_avg)


def load_untrained_SIR_model(PARAMS):
    """Load an untrained model (with weights initiatted) according to model paramateters in the 
    PARAMS dictionary

    Args:
        PARAMS (dict): dictionary of model/training parameters: i.e. hyperparameters and training parameters.

    Returns:
        utils.RegularizedRegressionModel object
    """
    model = SIR_Model(
        nfeatures=PARAMS['NFEATURES'],
        ntargets=1,
        nlayers=PARAMS["n_layers"],
        hidden_size=PARAMS["hidden_size"],
        dropout=PARAMS["dropout"],
        # activation=PARAMS["activation"]
    )
    # model.apply(initialize_weights)
    print('INITIATED UNTRAINED MODEL:',
          # model
         )
    # print(model)
    return model
    
 
SIR_Model_Huber_PARAMS = {
"n_layers": int(5),
"hidden_size": int(10),
"dropout": float(0.13),
"NFEATURES":int(N_Features),
"activation": "SiLU",
'optimizer_name':'NAdam',
    # 'optimizer_name':'RMSprop',
'starting_learning_rate':float(0.00003),
'momentum':float(0.9),
'batch_size':int(60),
'n_iterations': int(3e6),
'traces_step':int(100),
'L2':float(0.1),
'pth_string':'SIR_SILU_Huber_loss.pth'
}
   
   
untrained_SIR_model_Huber = load_untrained_SIR_model(SIR_Model_Huber_PARAMS)

BATCHSIZE=SIR_Model_Huber_PARAMS["batch_size"]
traces_SIR = ([], [], [], [])
traces_step = 1000
optimizer_name=SIR_Model_Huber_PARAMS["optimizer_name"]


optimizer_SIR = getattr(torch.optim, str(optimizer_name))(untrained_SIR_model_Huber.parameters(), 
                                                           lr=SIR_Model_Huber_PARAMS["starting_learning_rate"])

traces_SIR = train_SIR(model=untrained_SIR_model_Huber, 
              optimizer=optimizer_SIR, 
              # avloss=average_quadratic_loss,
                            avloss=Huber_loss,
              batch_size=BATCHSIZE, 
              n_iterations=SIR_Model_Huber_PARAMS["n_iterations"], 
              traces=traces_SIR, 
              step=traces_step, 
              window=200,
                      PARAMS=SIR_Model_Huber_PARAMS,
                      pth_string = SIR_Model_Huber_PARAMS['pth_string'])
