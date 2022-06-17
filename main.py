import torch
import torch.nn as nn
import fisher

network = nn.Sequential(
				nn.Linear(15,4),
				nn.Tanh(),
				nn.Linear(4,3),
				nn.LogSoftmax(dim=1)
	)
epsilon = 1e-3
input_data = torch.randn((1,15))
network.zero_grad()
output = network(input_data).max()
output.backward()
vector = []
for parameter in network.parameters():
	vector.append(parameter.grad.clone())

FISHER = fisher.FISHER_OPERATION(input_data, network, vector, epsilon)
print("The fisher matrix quadratic form:", FISHER.fisher_quadratic_form())
print("The fisher matrix trace:", FISHER.fisher_trace())
print("The fisher information sensitivity:", FISHER.fisher_sensitivity())
