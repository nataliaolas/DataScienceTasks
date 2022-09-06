#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import matplotlib.pyplot as plt


# In[54]:


zoo = pandas.read_csv("/Users/Komputer/Downloads/zoo.csv", names=["nazwa","wlosy","pierze","jajka","mleko","latajacy","wodny","drapieznik","uzebiony",
"szkielet","oddycha","jadowity","pletwy","nogi","ogon","udomowiony","rozmiar_kota","typ"])


# In[3]:


class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50]])
        # self.weights = np.full((15,1), 0.50)
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


# In[4]:


### 1.1) Preparing datas:
inputs = np.loadtxt("/Users/Komputer/Downloads/zoo.csv", delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))

### Normilization of input data
normMoja = inputs.max()
inputs = inputs / normMoja
print( inputs.shape )
print(inputs)


# In[5]:


## 2) Creation of a neural network and training
################################################################################
outputs_temp = np.loadtxt("/Users/Komputer/Downloads/zoo.csv", delimiter=',', usecols=17, dtype='|S15', ndmin=2)
outputs = (outputs_temp == b'4').astype(int)
# # tworzenie sieci neuronowej
NN = NeuralNetwork(inputs, outputs)
# # uczenie sieci neuronowej
NN.train(100)


# In[6]:


## 3) Using a neural network to determine whether a given object is an object of a specific class of target variable
zaskroniec = np.array( [0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0])
pepe = np.array([1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1])
bass = np.array([0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0])
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(NN.predict(zaskroniec), ' - Powinno byc: ', 0)
print(NN.predict(pepe), ' - Powinno byc: ', 0)
bass_output = NN.predict(bass)
print(bass_output, ' - Powinno byc: ', 1)


# In[7]:


licznik = 0
licznik_bledow = 0
for data,output in zip(inputs,outputs):
    predict_output = NN.predict(data)
    print(predict_output, "powinno byc", output)
    licznik += 1
    if round(predict_output[0],0) != output:
        licznik_bledow += 1


# In[8]:


plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()


# In[9]:


print(NN.weights)


# In[10]:


print(NN.error)


# In[11]:


print("Poprawnie zakwalifikowanych: ", (licznik-licznik_bledow))
print("Blednie zakwalifikowanych: ", licznik_bledow)


# Ex 2:

# In[12]:


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


# In[13]:


inputs = np.loadtxt("/Users/Komputer/Downloads/zoo.csv", delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))


# In[14]:


normMoja = inputs.max()
inputs = inputs / normMoja


# In[15]:


outputs_temp = np.loadtxt("/Users/Komputer/Downloads/zoo.csv", delimiter=',', usecols=17, dtype='|S15', ndmin=2)


# In[16]:


print(outputs_temp)


# In[17]:


_, outputs = np.unique(outputs_temp, return_inverse=True)


# In[18]:


outputs = outputs + 1
print(outputs)


# In[19]:


mlp = MLPClassifier(hidden_layer_sizes=(4,6), max_iter=20000, random_state=5)
mlp.fit(inputs, outputs)


# In[20]:


print("liczba warstw: ", mlp.n_layers_)
print("liczba iteracji: ", mlp.n_iter_)
print("liczba neuronow w warstwie wyjsciowej: ", mlp.n_outputs_)
print("wartosc funkcji loss:", mlp.loss_)
print("wartosci klas wynikowych:", mlp.classes_)
print("wartosci wag na polaczeniach:", mlp.coefs_)


# In[21]:


bledne_indeksy = []
indeks = 0
for data,output in zip(inputs,outputs):
    predict_output = mlp.predict([data])
    print(predict_output, "powinno byc", output)
    if predict_output != output:
        bledne_indeksy.append(indeks)
    indeks += 1


# In[22]:


poprawne_klasyfikacje = indeks - len(bledne_indeksy)
print("Bledne klasyfikacje, indeksy  wierszy:", bledne_indeksy)
print("Licznik poprawnych klasyfikacji:", poprawne_klasyfikacje)


# Zadanie 4:

# In[23]:


from SALib.sample import saltelli
from SALib.analyze import sobol


# In[24]:


problem = {'num_vars': 16,
           'names': ["wlosy","pierze","jajka","mleko","latajacy",
                        "wodny","drapieznik","uzebiony",
                        "szkielet","oddycha","jadowity","pletwy",
                        "nogi","ogon","udomowiony","rozmiar_kota"
                    ],
           'bounds': [[0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1],
                      [0, 1]
                      ]
           }


# In[25]:


param_values = saltelli.sample(problem, 2048, calc_second_order=True)
print(param_values.shape)

Y = mlp.predict(param_values)

Si = sobol.analyze(problem, Y, print_to_console=True)


# Ex 3:

# In[26]:


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


# In[27]:


wina = pandas.read_csv("/Users/Komputer/Downloads/winequality-white.csv")


# In[28]:


wina


# In[29]:


inputs = np.loadtxt("/Users/Komputer/Downloads/winequality-white.csv",delimiter=',',skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10))


# In[30]:


inputs


# In[31]:


normMoja = inputs.max()
inputs = inputs / normMoja


# In[32]:


outputs_temp = np.loadtxt("/Users/Komputer/Downloads/winequality-white.csv", delimiter=',',skiprows=1, usecols=11, dtype='|S15', ndmin=2)


# In[33]:


print(outputs_temp)


# In[34]:


_, outputs = np.unique(outputs_temp, return_inverse=True)


# In[35]:


outputs = outputs + 1
print(outputs)


# In[36]:


mlp = MLPClassifier(hidden_layer_sizes=(4,6), max_iter=20000, random_state=5)
mlp.fit(inputs, outputs)


# In[37]:


print("liczba warstw: ", mlp.n_layers_)
print("liczba iteracji: ", mlp.n_iter_)
print("liczba neuronow w warstwie wyjsciowej: ", mlp.n_outputs_)
print("wartosc funkcji loss:", mlp.loss_)
print("wartosci klas wynikowych:", mlp.classes_)
print("wartosci wag na polaczeniach:", mlp.coefs_)


# In[38]:


bledne_indeksy = []
indeks = 0
for data,output in zip(inputs,outputs):
    predict_output = mlp.predict([data])
    print(predict_output, "powinno byc", output)
    if predict_output != output:
        bledne_indeksy.append(indeks)
    indeks += 1


# In[39]:


poprawne_klasyfikacje = indeks - len(bledne_indeksy)
print("bledne klasyfikacje, indeksy  wierszy:", bledne_indeksy)
print("Licznik poprawnych klasyfikacji:", poprawne_klasyfikacje)



