import torch
import torch.nn.functional as F 
from copy import deepcopy


class FISHER_OPERATION(object):
	def __init__(self, input_data, network, vector, epsilon = 1e-3):
		self.input = input_data 
		self.network = network
		self.vector = vector
		self.epsilon = epsilon

	# Computes the fisher matrix quadratic form along the specific vector
	def fisher_quadratic_form(self):
		fisher_sum = 0
		## Computes the gradient of parameters of each layer
		for i, parameter in enumerate(self.network.parameters()):
			## Store the original parameters
			store_data = deepcopy(parameter.data)	
			parameter.data += self.epsilon * self.vector[i]
			log_softmax_output1 = self.network(self.input)
			softmax_output1 = F.softmax(log_softmax_output1, dim=1)
			parameter.data -= 2 * self.epsilon * self.vector[i]
			log_softmax_output2 = self.network(self.input)
			solfmax_output2 = F.softmax(log_softmax_output2, dim=1)
			parameter.data = store_data
			# The summation of finite difference approximate
			fisher_sum += (((log_softmax_output1 - log_softmax_output2)/(2 * self.epsilon))*((softmax_output1 - solfmax_output2)/(2 * self.epsilon))).sum()
		return fisher_sum 


	# Computes the fisher matrix trace
	def fisher_trace(self):
		fisher_trace = 0
		output = self.network(self.input)
		output_dim = output.shape[1]
		parameters = self.network.parameters()
		## Computes the gradient of parameters of each layer
		for parameter in parameters:
			for j in range(output_dim):
				self.network.zero_grad()
				log_softmax_output = self.network(self.input)
				log_softmax_output[0,j].backward()
				log_softmax_grad = parameter.grad
				self.network.zero_grad()
				softmax_output = F.softmax(self.network(self.input), dim=1)
				softmax_output[0,j].backward()
				softmax_grad = parameter.grad
				fisher_trace += (log_softmax_grad * softmax_grad).sum()
		return fisher_trace


	# Computes fisher information sensitivity for x and v.
	def fisher_sensitivity(self):
		output = self.network(self.input)
		output_dim = output.shape[1]
		parameters = self.network.parameters()
		x = deepcopy(self.input.data)
		x.requires_grad = True
		fisher_sum = 0
		for i, parameter in enumerate(parameters):
			for j in range(output_dim):				
				store_data = deepcopy(parameter.data)
				# plus eps
				parameter.data += self.epsilon * self.vector[i]
				log_softmax_output1 = self.network(x)
				log_softmax_output1[0,j].backward()
				new_plus_log_softmax_grad = deepcopy(x.grad.data)
				x.grad.zero_()
				self.network.zero_grad()
				softmax_output1 = F.softmax(self.network(x), dim=1)
				softmax_output1[0,j].backward()	
				new_plus_softmax_grad = deepcopy(x.grad.data)
				x.grad.zero_()
				self.network.zero_grad() 			
				# minus eps
				parameter.data -= 2 * self.epsilon * self.vector[i]
				log_softmax_output2 = self.network(x)
				log_softmax_output2[0,j].backward()
				new_minus_log_softmax_grad = deepcopy(x.grad.data)
				x.grad.zero_()
				self.network.zero_grad()
				softmax_output2 = F.softmax(self.network(x), dim=1)
				softmax_output2[0,j].backward()
				new_minus_softmax_grad = deepcopy(x.grad.data)
				x.grad.zero_()
				self.network.zero_grad()
				# reset and evaluate
				parameter.data = store_data
				fisher_sum += 1/(2 * self.epsilon)**2 * ((new_plus_log_softmax_grad - new_minus_log_softmax_grad)*(new_plus_softmax_grad - new_minus_softmax_grad))
		return fisher_sum

