from __future__ import annotations
import torch 
import torch.nn as nn
class TrajectoryRNN(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,num_layers:int,output_size:int,rnn_type:str="lstm",dropout:float=0.0)->None:
        super().__init__()
        self.input_size=input_size #nombre de features d'entrée 
        self.hidden_size=hidden_size #taille de l'état caché 
        self.num_layers=num_layers #nombre de couches RNN
        self.output_size=output_size #taille de la sortie
        self.rnn_type=rnn_type.lower() #type de RNN (lstm ou gru)
        self.dropout=dropout #taux de dropout
        rnn_kwargs=dict(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout)
        if self.rnn_type=="lstm":
            self.rnn=nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Invalid rnn_type: {self.rnn_type}")
        self.head=nn.Sequential(nn.Linear(hidden_size,hidden_size//2),nn.ReLU(),nn.Linear(hidden_size//2,output_size))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        rnn_out,_=self.rnn(x) #rnn_out:(batch_size,seq_len,hidden_size)
        last_step=rnn_out[:,-1,:]
        return self.head(last_step)
    def get_config(self) -> dict:
       
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "rnn_type": self.rnn_type,
        }