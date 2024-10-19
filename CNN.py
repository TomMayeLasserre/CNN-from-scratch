import numpy as np


class CNN:
    def __init__(self, input_shape, conv_params, hidden_sizes, output_size):
        # Paramètres de convolution
        self.num_filters = conv_params['num_filters']
        self.filter_size = conv_params['filter_size']
        self.stride = conv_params.get('stride', 1)
        self.pad = conv_params.get('pad', 0)

        # Initialisation des filtres de convolution
        self.filters = np.random.randn(
            self.num_filters, self.filter_size, self.filter_size) / (self.filter_size * self.filter_size)
        self.conv_biases = np.zeros((self.num_filters, 1))

        # Calcul de la taille de la sortie après convolution
        self.conv_output_shape = (
            (input_shape[0] - self.filter_size +
             2 * self.pad) // self.stride + 1,
            (input_shape[1] - self.filter_size +
             2 * self.pad) // self.stride + 1
        )

        # Paramètres de pooling
        self.pool_size = conv_params.get('pool_size', 2)
        self.pool_stride = conv_params.get('pool_stride', 2)

        # Calcul de la taille de la sortie après pooling
        self.pool_output_shape = (
            (self.conv_output_shape[0] -
             self.pool_size) // self.pool_stride + 1,
            (self.conv_output_shape[1] -
             self.pool_size) // self.pool_stride + 1
        )

        # Taille de l'entrée pour les couches entièrement connectées
        fc_input_size = self.num_filters * \
            self.pool_output_shape[0] * self.pool_output_shape[1]

        # Initialisation des poids pour les couches entièrement connectées
        layer_sizes = [fc_input_size] + hidden_sizes + [output_size]
        self.fc_layers = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(
                layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.fc_layers.append({'weight': weight, 'bias': bias})

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def pad_input(self, x):
        if self.pad == 0:
            return x
        else:
            return np.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

    def convolution(self, x):
        batch_size, height, width = x.shape
        conv_height, conv_width = self.conv_output_shape
        conv_output = np.zeros(
            (batch_size, self.num_filters, conv_height, conv_width))

        x_padded = self.pad_input(x)

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(conv_height):
                    for j in range(conv_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size

                        region = x_padded[b, h_start:h_end, w_start:w_end]
                        conv_output[b, f, i, j] = np.sum(
                            region * self.filters[f]) + self.conv_biases[f]
        return conv_output

    def convolution_backward(self, dconv_prev, x):
        batch_size, height, width = x.shape
        x_padded = self.pad_input(x)
        dfilters = np.zeros_like(self.filters)
        dbiases = np.zeros_like(self.conv_biases)
        dx_padded = np.zeros_like(x_padded)

        conv_height, conv_width = self.conv_output_shape

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(conv_height):
                    for j in range(conv_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size

                        region = x_padded[b, h_start:h_end, w_start:w_end]

                        dfilters[f] += dconv_prev[b, f, i, j] * region
                        dbiases[f] += dconv_prev[b, f, i, j]
                        dx_padded[b, h_start:h_end, w_start:w_end] += dconv_prev[b,
                                                                                 f, i, j] * self.filters[f]

        if self.pad != 0:
            dx = dx_padded[:, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dx = dx_padded
        return dx, dfilters, dbiases

    def max_pool(self, x):
        batch_size, num_filters, height, width = x.shape
        pool_height, pool_width = self.pool_output_shape
        pooled = np.zeros((batch_size, num_filters, pool_height, pool_width))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(pool_height):
                    for j in range(pool_width):
                        h_start = i * self.pool_stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.pool_stride
                        w_end = w_start + self.pool_size

                        region = x[b, f, h_start:h_end, w_start:w_end]
                        pooled[b, f, i, j] = np.max(region)
        return pooled

    def max_pool_backward(self, dpool, x):
        batch_size, num_filters, height, width = x.shape
        dx = np.zeros_like(x)
        pool_height, pool_width = self.pool_output_shape

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(pool_height):
                    for j in range(pool_width):
                        h_start = i * self.pool_stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.pool_stride
                        w_end = w_start + self.pool_size

                        region = x[b, f, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        for m in range(h_start, h_end):
                            for n in range(w_start, w_end):
                                if x[b, f, m, n] == max_val:
                                    dx[b, f, m, n] += dpool[b, f, i, j]
        return dx

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def flatten_backward(self, dflattened, x_shape):
        return dflattened.reshape(x_shape)

    def forward(self, x):
        self.x_input = x
        # Convolutional Layer
        self.conv_output = self.convolution(x)
        self.conv_activated = self.relu(self.conv_output)

        # Pooling Layer
        self.pooled_output = self.max_pool(self.conv_activated)

        # Flatten
        self.flattened = self.flatten(self.pooled_output)

        # Fully Connected Layers
        activations = [self.flattened]
        input = self.flattened

        for layer in self.fc_layers[:-1]:
            z = np.dot(input, layer['weight']) + layer['bias']
            input = self.relu(z)
            activations.append(input)

        # Output Layer with Softmax
        z = np.dot(input, self.fc_layers[-1]
                   ['weight']) + self.fc_layers[-1]['bias']
        output = self.softmax(z)
        activations.append(output)

        return activations

    def backward(self, activations, y_true, learning_rate):
        batch_size = y_true.shape[0]
        # Erreur à la couche de sortie
        delta = (activations[-1] - y_true) / batch_size

        # Mise à jour des couches entièrement connectées
        for i in reversed(range(len(self.fc_layers))):
            a_prev = activations[i]
            dw = np.dot(a_prev.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            self.fc_layers[i]['weight'] -= learning_rate * dw
            self.fc_layers[i]['bias'] -= learning_rate * db

            if i != 0:
                delta = np.dot(
                    delta, self.fc_layers[i]['weight'].T) * self.relu_derivative(activations[i])
            else:
                # Pas de fonction d'activation après l'aplatissement
                delta = np.dot(delta, self.fc_layers[i]['weight'].T)

        # Rétropropagation à travers la couche d'aplatissement
        dflatten = delta

        # Remodelage pour correspondre à la sortie du pooling
        dpool = self.flatten_backward(dflatten, self.pooled_output.shape)

        # Rétropropagation à travers le pooling
        dconv_relu = self.max_pool_backward(dpool, self.conv_activated)

        # Rétropropagation à travers ReLU
        dconv = dconv_relu * self.relu_derivative(self.conv_output)

        # Rétropropagation à travers la convolution
        dx, dfilters, dbiases = self.convolution_backward(dconv, self.x_input)

        # Mise à jour des filtres de convolution et des biais
        self.filters -= learning_rate * dfilters
        self.conv_biases -= learning_rate * dbiases

    def train(self, x_train, y_train, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            activations = self.forward(x_train)
            self.backward(activations, y_train, learning_rate)

            if epoch % 1 == 0:
                loss = - \
                    np.sum(
                        y_train * np.log(activations[-1] + 1e-9)) / y_train.shape[0]
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
                losses.append(loss)
        return losses

    def predict(self, x):
        activations = self.forward(x)
        return activations[-1]
