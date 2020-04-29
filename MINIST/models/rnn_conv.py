import torch.nn as nn
import torch
class ImageRNN(nn.Module):
	def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, device):
		super(ImageRNN, self).__init__()
		self.device = device
		self.n_neurons = n_neurons   # Hidden layer (neurons)
		self.batch_size = batch_size
		self.n_steps = n_steps    # 64
		self.n_inputs = n_inputs  # 28
		self.n_outputs = n_outputs  # 10
		# Basic RNN layer
		self.rnn = nn.LSTM(n_inputs, n_neurons, 1, batch_first=True) #apparently this line was the cause of the bad results
		# Followed by a fully connected layer
		self.FC = nn.Linear(self.n_neurons, self.n_outputs)

	def init_hidden(self, ):
		# (num_layers, batch_size, n_neurons)
		return (torch.zeros(1, self.batch_size, self.n_neurons)).to(self.device)

	def forward(self, x):
		h0 = torch.zeros(1, x.size(0), self.n_neurons).to(self.device)
		c0 = torch.zeros(1, x.size(0), self.n_neurons).to(self.device)
		out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = self.FC(out[:, -1, :])
		return out