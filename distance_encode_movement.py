#train the network by classifying windows from the different biking styles
import motion_data as mmd
import numpy as np
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")

skip_windows=5
window_length=50
class MovementClass(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X,Y):
        #generate X and Y
        self.X = X
        self.Y=Y
        self.length = len(self.X)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]
def generate_windows_and_labels(motions_data,window_lengt=60):
    #generate sliding window data and divide in batches
    X = []
    Y = []
    for m_idx, motion_data in enumerate(motions_data):
        start_frames = list(range(window_length, len(motion_data), skip_windows))
        windows = np.array([motion_data[i - window_length:i] for i in start_frames])
        # format is samples,window,feature
        windows = windows.transpose((0, 2, 1))
        X.extend(windows)
        Y.extend(np.ones(len(windows)) * m_idx)
    return X,Y



class EncDec(nn.Module):
    #kernel size is proposed to be half a second long
    def __init__(self,motion_feature_size,n_classes,hidden_size=10,kernel_size=10,dropout=0.5):
        super(EncDec, self).__init__()
        self.dropout=dropout
        #enc
        self.conv1 = nn.Conv1d(motion_feature_size, hidden_size, kernel_size)
        self.batchnorm1=nn.BatchNorm1d(hidden_size)
        self.FC_mu = nn.Linear(hidden_size,hidden_size)
        self.FC_log_sigma = nn.Linear(hidden_size, hidden_size)
        #dec
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.deconv1 = nn.ConvTranspose1d(hidden_size, motion_feature_size, kernel_size=kernel_size )
        #class
        self.fcn = nn.Linear(hidden_size, n_classes)

    def encode(self,x):
        x_len = x.size(0)
        x = self.conv1(x)
        kernel_size = x.size()[2]
        x, idx1 = F.max_pool1d(x, stride=1, kernel_size=kernel_size, return_indices=True)
        x = F.tanh(x)
        hidden = x.view(x_len, -1)
        return hidden
    def forward(self, x):
        x_len = x.size(0)
        #variational enc part
        x=self.conv1(x)
        x=F.relu(x)
        x=self.batchnorm1(x)
        x=F.dropout(x,p=self.dropout)
        size1 = x.size()
        kernel_size=x.size()[2]
        x,idx1=F.max_pool1d(x,stride=1,kernel_size=kernel_size,return_indices=True)
        x = x.view(x_len, -1)
        mu=self.FC_mu(x)
        log_sigma = self.FC_log_sigma(x)
        x=self._sample(mu,log_sigma)
        hidden=x
        #dec part
        x=x.unsqueeze(2)
        x = F.max_unpool1d(x, idx1,stride=1,kernel_size=kernel_size,output_size=size1)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.deconv1(x)

        #class part
        y = F.dropout(hidden, p=self.dropout)
        y = self.fcn(y)
        return x,y, hidden
    def _sample(self,mu,log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            self.mu=mu
            self.log_sigma=log_sigma
            #reparamitrisation trick
            return eps.mul(std).add_(mu)
        else:
            return mu
    @staticmethod
    #KL divergence
    def latent_loss(mu, log_sigma):
        return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())


#the loss should promote as high variance as possible
def variation_loss(hidden):
    #reward variance
    return -torch.mean(torch.var(hidden,dim=0))
def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov
def cross_corr_loss(hidden):
    corr_h=cov(hidden)/(torch.std(hidden)*torch.std(hidden))
    return torch.mean(torch.ones(corr_h.size()).to(device)-corr_h)
disentangled_VAE_loss_weight=5
class_loss_weight=1
cross_corr_loss_weight=0
reconstruction_loss_weight=1
n_epochs = 1000

hidden_size=10
def train_mov_enc(X,Y,model_name="mov_dist_enc"):
    # input format N,C,L
    # C is the sensor features (19)
    print(np.shape(X)[1])
    model=EncDec(motion_feature_size=np.shape(X)[1],n_classes=len(set(Y)),hidden_size=hidden_size)
    model.to(device)
    print("#params: ",np.sum([np.prod(par.size()) for par in (model.parameters())]))

    criterion = nn.MSELoss()

    class_criterion=nn.CrossEntropyLoss()
    model_opt = optim.SGD(model.parameters(), lr=0.001,weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(model_opt,T_max=100)

    dataset = MovementClass(X,Y)

    validation_idx = list(np.random.choice(list(range(len(X))), int(len(X) / 5), replace=False))
    train_idx = list(set(range(len(X))) - set(validation_idx))


    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    XY_DL_training = DataLoader(dataset, batch_size=100, sampler=train_sampler)
    XY_DL_validation = DataLoader(dataset, batch_size=100, sampler=validation_sampler)


    ### training
    early_stop_ep = 10
    early_stop_loss_gain = 0.00001
    early_stop_loss=200
    disentangled_VAE_loss=5
    class_loss_weight=1
    cross_corr_loss_weight=0
    reconstruction_loss_weight=1
    for n in range(n_epochs):
        print("epoch: ", n)
        training_loss = []
        for i, (X_batch,Y_batch) in enumerate(XY_DL_training):
            X_batch = torch.tensor(X_batch, dtype=torch.float, device=device)
            Y_batch = torch.tensor(Y_batch, dtype=torch.long, device=device)
            X_pred,Y_pred,hidden= model(X_batch)
            class_loss=class_criterion(Y_pred,Y_batch)
            loss = reconstruction_loss_weight*criterion(X_pred, X_batch) + \
                   disentangled_VAE_loss * EncDec.latent_loss(model.mu,model.log_sigma) + \
                   class_loss_weight*class_loss + \
                   cross_corr_loss_weight*cross_corr_loss(hidden)
            training_loss.append(loss.item())
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
        vall_loss=[]
        for i, (X_batch, Y_batch) in enumerate(XY_DL_validation):
            X_batch = torch.tensor(X_batch, dtype=torch.float, device=device)
            Y_batch = torch.tensor(Y_batch, dtype=torch.long, device=device)
            X_pred,Y_pred,hidden = model(X_batch)
            class_loss = class_criterion(Y_pred, Y_batch)
            loss = reconstruction_loss_weight * criterion(X_pred, X_batch) + \
                   disentangled_VAE_loss * EncDec.latent_loss(model.mu, model.log_sigma) + \
                   class_loss_weight * class_loss + \
                   cross_corr_loss_weight * cross_corr_loss(hidden)
            vall_loss.append(loss.item())
        print("train loss", np.mean(training_loss))
        print("test loss", np.mean(vall_loss) )
        # early stop
        if n % early_stop_ep == 0:
            early_stop_loss = min(np.mean(vall_loss), early_stop_loss)
        # quit training if after n epochs the los is above the max gain, stop
        if n % early_stop_ep == early_stop_ep - 1 and early_stop_loss -np.mean(vall_loss) < early_stop_loss_gain:
            print("early stop: ", n)
            break
    import datetime
    start_time = datetime.datetime.now()
    model.to("cpu")
    torch.save(model.state_dict(),model_name+"_"+str(start_time)+".pth")
    return model
def main():
    # Train
    motion_loading=mmd.motion_load_qualisys_tsv
    motion_names=mmd.bike_names
    motion_dict=dict([(motion_name,motion_loading(motion_name,skip_rows=10,skip_columns=2)) for motion_name in motion_names])
    motions_data =[mmd.center_norm_data(md[0]) for md in motion_dict.values()]
    motions_data=[md.reshape(md.shape[0],-1) for md in motions_data]
    X,Y=generate_windows_and_labels(motions_data)
    model=train_mov_enc(X,Y,"dist_mov_enc_bike")

    # test the encoding

    # model=EncDec(motion_feature_size=np.shape(X[0][0])[1])
    #
    # model.load("mov_enc_bike.pickle")
    # X= np.concatenate([X[m] for m in range(len(X))])
    # for X_batch in X[:3]:
    #     X_batch = torch.tensor(X_batch, dtype=torch.float, device=device)
    #     #convert 1 batch
    #     hidden=model.encode(X_batch)
    #     hist,_=np.histogram(hidden[0].data.cpu(), normed=True)
    #     print(entropy(hist), entropy([1/len(hist)]*len(hist)))



if __name__ == "__main__": main()


