import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(sigmoid_x):
    return sigmoid_x * (1.0 - sigmoid_x)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)


def show_result(X, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='bwr')

    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)
    plt.scatter(X[:, 0], X[:, 1], c=pred_y[:, 0], cmap='bwr')
    
    plt.savefig('res.png')
    plt.show()
    

class NN:
    weights = [] # starts between layer 1 and layer 2
    biases = [] # starts from the second layer
    layers = [] # pre activation
    activated = []

    # adam stuff
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 0
    m_weights = []
    v_weights = []
    m_biases = []
    v_biases = []

    def __init__(self, X, y, layers, model_name, activation, optim):
        self.layer_count = len(layers)
        self.X = X
        self.y = y
        self.num_samples = y.shape[0]
        self.model_name = model_name
        self.activation = activation
        self.optim = optim
        
        for i in range(self.layer_count - 1):
            self.weights.append(self.xavier_init(layers[i], layers[i + 1]))
            self.biases.append(np.zeros(layers[i + 1]))

            self.m_weights.append(np.zeros((layers[i], layers[i + 1])))
            self.v_weights.append(np.zeros((layers[i], layers[i + 1])))
            self.m_biases.append(np.zeros(layers[i + 1]))
            self.v_biases.append(np.zeros(layers[i + 1]))

    def xavier_init(self, node_prev, node_next):
        upper = (np.sqrt(6.0) / np.sqrt(node_prev + node_next))
        lower = -upper
        return np.random.uniform(lower, upper, (node_prev, node_next))
    
    def forward(self):
        self.layers.clear()
        self.activated.clear()

        current_layer = self.X.copy()
        for i in range(self.layer_count - 1):
            z = current_layer.dot(self.weights[i]) + self.biases[i]
            self.layers.append(z)
            
            if self.activation == "sigmoid":
                current_layer = sigmoid(z) 
            elif self.activation == "relu":
                current_layer = relu(z)
            elif self.activation == "none":
                current_layer = z
            else:
                raise Exception("Invalid activation function setting")
                
            self.activated.append(current_layer)
    
    def backward(self, lr):
        dLdA = (self.activated[-1] - self.y) / self.num_samples

        d_weights = [None] * self.layer_count
        d_biases = [None] * self.layer_count

        for i in reversed(range(self.layer_count - 1)):
            if self.activation == "sigmoid":
                dLdZ = dLdA * d_sigmoid(self.activated[i]) 
            elif self.activation == "relu":
                dLdZ = dLdA * d_relu(self.layers[i])
            elif self.activation == "none":
                dLdZ = dLdA
            else:
                raise Exception("Invalid activation function setting")

            d_weights[i] = (self.activated[i - 1].T.dot(dLdZ)) if i > 0 else (self.X.T.dot(dLdZ))
            d_biases[i] = np.sum(dLdZ, axis=0)

            if i > 0:
                dLdA = dLdZ.dot(self.weights[i].T)

        if self.optim == "adam":
            self.t += 1
            for i in range(self.layer_count - 1):
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * d_weights[i]
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * d_biases[i]

                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (d_weights[i] ** 2)
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (d_biases[i] ** 2)

                m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)

                m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)

                self.weights[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        elif self.optim == "sgd":
            for i in range(self.layer_count - 1):
                self.weights[i] -= lr * d_weights[i]
                self.biases[i] -= lr * d_biases[i]
        else:
            raise Exception("Invalid optimizer setting")

    def train(self, epochs=10000, lr=0.1, lr_scheduling=False):
        losses = []
        for epoch in range(1, epochs + 1):
            self.forward()
            self.backward(lr)
            
            if lr_scheduling and epoch % (epochs / 5) == 0:
                lr = lr * 0.7
            
            loss = np.mean((self.activated[-1] - self.y) ** 2)
            losses.append(loss)
            if epoch % 1000 == 0 or epoch == 1:
                print(f"epoch {epoch}: loss = {loss}")
                
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.xlabel("Loss")
        plt.savefig("loss_epoch.png")
        
        with open(self.model_name, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases, "activation": self.activation}, f)
    
    def load(self):
        with open(self.model_name, "rb") as f:
            model = pickle.load(f)
            self.weights = model["weights"]
            self.biases = model["biases"]
            self.activation = model["activation"]
            self.layer_count = len(self.weights)

    def predict_prob(self, x):
        current_layer = x
        for i in range(self.layer_count):
            z = current_layer.dot(self.weights[i]) + self.biases[i]
            if self.activation == "sigmoid":
                current_layer = sigmoid(z)
            elif self.activation == "relu":
                current_layer = relu(z)
            elif self.activation == "none":
                current_layer = z
        return current_layer


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, choices=['linear', 'xor'], required=True, help='linear / xor configuration')
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='train / test mode')
parser.add_argument('--activation', type=str, choices=['sigmoid', 'relu', 'none'], default='sigmoid', help='activation function to use')
parser.add_argument('--graph', action='store_true', help='graph predictions')

args = parser.parse_args()

mode = args.mode        # train / test
config = args.config    # linear / xor
graph = args.graph    # true / false
activation = args.activation # sigmoid / relu

print(f"Mode: {mode}")
print(f"Config: {config}")

if config == "linear":
    # Config for Linear
    if mode == "train":
        X, y = generate_linear(n=1000)
    else:
        X, y = generate_linear()
    nn = NN(X, y, [2, 4, 4, 1], model_name="model_linear.pkl", activation=activation, optim="adam")

    if mode == "train":
        nn.train(5000, 0.1, lr_scheduling=False)

elif config == "xor":
    # Config for XOR
    X, y = generate_XOR_easy()
    nn = NN(X, y, [2, 4, 4, 1], model_name="model_xor.pkl", activation=activation, optim="adam")

    if mode == "train":
        nn.train(5000, 0.1)

nn.load()
predicted_prob = nn.predict_prob(X)
error = np.mean((y - predicted_prob) ** 2)
prediction = (predicted_prob > 0.5).astype(int)

for i in range(len(y)):
    print(f"iter {(i + 1):2d} |\tGround Truth: {y[i][0]} |\tprediction: {prediction[i][0]} ({predicted_prob[i][0]:.6f}) |")

print("prediction loss: " if mode == "test" else "training loss: ", error)
print("prediction accuracy: " if mode == "test" else "training accuracy", 1 - np.abs(np.sum(prediction - y)) / len(y))
    
if graph:
    show_result(X, y, prediction)