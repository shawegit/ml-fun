import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grad_sig(x):
    return x * (1 - x)


def relu(x):
    return np.max(x, 0)


def grad_relu(x):
    return 1 if x > 0 else 0


class Neuron:
    l_rate = 0.5

    def __init__(self, dim, func=relu, grad_func=grad_relu, init_weights=None):
        self.weights = np.random.rand(dim) if init_weights is None else np.array(init_weights)
        self.activation_func = func
        self.grad_func = grad_func
        self._last_out = 0

    def compute(self, input, bias):
        self._last_out = self.activation_func(np.dot(input, self.weights) + bias)

    def update_weights(self, error, prev):
        updater = error * self.grad_func(self.last_out)
        error_update = updater * self.weights
        self.weights -= Neuron.l_rate * updater * np.array(prev)
        return error_update

    @property
    def last_out(self):
        return self._last_out

    @last_out.setter
    def last_out(self, _):
        raise Exception


class InputLayer:
    def __init__(self, value):
        self._output = value

    @property
    def last_out(self):
        return self._output

    @last_out.setter
    def last_out(self, _):
        raise Exception


def forward_pass(net, biases):
    for in_lay, layer, b in zip(net, net[1:], biases):
        neuron_in = [v.last_out for v in in_lay]
        for neuron in layer:
            neuron.compute(neuron_in, b)


def backward_pass(net, errors):
    for layer, prev_layer in zip(reversed(net), reversed(net[:-1])):
        prev = [neuron.last_out for neuron in prev_layer]
        new_error = np.zeros_like(prev)
        for error, neuron in zip(errors, layer):
            new_error += neuron.update_weights(error, prev)
        errors = new_error


def train(net, labels, n_iter):
    for _ in range(n_iter):
        forward_pass(net, bias)
        errors = [neuron.last_out - label for label, neuron in zip(labels, net[-1])]
        print(.5 * np.sum(errors) ** 2)
        backward_pass(net, errors)


if __name__ == '__main__':
    input_layer = [InputLayer(.05), InputLayer(.1)]
    labels = [.01, .99]
    bias = [.0, .0, .0]
    net_structure = [2, 5, len(labels)]

    sz = len(input_layer)
    network = [input_layer]
    for n_neurons in net_structure:
        network.append([Neuron(sz) for _ in range(n_neurons)])
        sz = n_neurons

    train(network, labels, n_iter=100)
    print([neuron.last_out for neuron in network[-1]])
'''
new_error = np.zeros(2)
for e, neuron in zip(error, layers[-1]):
    himpf = e * neuron.last_out*(1 - neuron.last_out)
    grad = himpf*np.array([layers[-2][0].last_out, layers[-2][1].last_out])
    new_error += himpf*neuron.weights
    neuron.weights -= l_rate*grad
print(new_error)

for e, neuron in zip(new_error, layers[-2]):
    himpf = e * neuron.last_out*(1 - neuron.last_out)
    grad = himpf*np.array(x)
    new_error += himpf*neuron.weights
    neuron.weights -= l_rate*grad
    print(neuron.weights)
'''
