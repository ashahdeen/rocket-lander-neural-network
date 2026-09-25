#importing all the necessary libraries
import csv
import random
import math
from typing import List,Dict


#Data Cleaning
#removing duplicates
def clean_data(raw_data :List[list[float]]) -> List[List[float]]:
    unique_entries = {}

    for row in raw_data:
        input_x = tuple(row[:2])
        unique_entries[input_x] = row

    return list(unique_entries.values())


#loading the csv file
def load_data(csv_file_path):
    raw_data =[]
    with open(csv_file_path,"r") as f:
        reader = csv.reader(f)
        next(reader,None)
        for row in reader:
            try:
                cleaned_row=[float(value.strip())for value in row]
                raw_data.append(cleaned_row)
            except ValueError:
                continue
        


    

    return clean_data(raw_data)

#Scaling our data

def min_max(data):
    #using the first rows of the data to initialize the minimum and maximum values
    num_features = len(data[0])
    min_v = data[0][:]
    max_v = data[0][:]
    #looping over every other row to look for the minimum and maximu values
    for row in data[1:]:
        for i in range(num_features):
            if row[i] < min_v[i]:
                min_v[i] = row[i]
            if row[i] > max_v[i]:
                max_v[i] = row[i]
    return min_v,max_v

def scaled_data(data,min_v,max_v):
    scaled = []
    for row in data:
        scaled_row =[]
        #looping through the row and if the minimum and maximum value are the same, it is set to zero
        #otherwise we scale it.
        for i in range(len(row)):
            if max_v[i] ==min_v[i]:
                sc_value = 0
            else:
                sc_value =(row[i]- min_v[i])/(max_v[i]- min_v[i])
            scaled_row.append(sc_value) 
        scaled.append(scaled_row) 
    return scaled

#saving the split
def save_split(filename: str, X: List[List[float]], Y: List[List[float]]):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for x, y in zip(X, Y):
            writer.writerow(list(x) + list(y))

#loading our file
csv_file_path= "/Users/ashadeen/Downloads/lander_game/ce889_dataCollection copy.csv"
lander_data= load_data(csv_file_path)

#separating out data into inputs and outputs
X_list =[row[:2] for row in lander_data]
Y_list =[row[2:] for row in lander_data]



#splitting our data
#combining our data and shuffling it and determining our split boundaries
data = list(zip(X_list,Y_list))
random.seed(42)
random.shuffle(data)

tot = len(data)
train_t = int(0.70 * tot)
val_t = int(0.85 * tot)

#splitting our data in train,test and validation sets
train_set= data[:train_t]
val_set= data[train_t:val_t]
test_set= data[val_t:]

# saving our test,val and train splits
def save_split(filename, X, Y):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for x, y in zip(X, Y):
            writer.writerow(list(x) + list(y))

#splitting our data into train,val and test splits
X_train = [x for x, _ in train_set]
Y_train = [y for _, y in train_set]

X_val = [x for x, _ in val_set]
Y_val = [y for _, y in val_set]

X_test = [x for x, _ in test_set]
Y_test = [y for _, y in test_set]

#scaling our train sets,by computing  min and max values from the training data
X_min, X_max = min_max(X_train)
Y_min, Y_max = min_max(Y_train)

X_train_scaled = scaled_data(X_train, X_min, X_max)
Y_train_scaled = scaled_data(Y_train, Y_min, Y_max)

#scaling our validation and test sets using min max already computed
X_val_scaled = scaled_data(X_val, X_min, X_max)
Y_val_scaled = scaled_data(Y_val, Y_min, Y_max)

X_test_scaled = scaled_data(X_test, X_min, X_max)
Y_test_scaled = scaled_data(Y_test, Y_min, Y_max)

#saving the train,test and validation sets
save_split("train_split.csv", X_train_scaled, Y_train_scaled)
save_split("val_split.csv", X_val_scaled, Y_val_scaled)
save_split("test_split.csv", X_test_scaled, Y_test_scaled)


#initializing weights,bias,sigmoid function ,sigmoid derivative and the feedforward
class Neuron:
    def __init__(self,num_inputs):

        #initializing weights and bias
        self.weights = [random.uniform(-1,1)for _ in range(num_inputs)]
        self.bias = random.uniform(-1,1)
        
        #storing the output and input
        self.l_output= 0.0
        self.l_input = None 

        #storing weights updates, this weights are updated after running backpropagation
        self.prev_weights = [0.0 for _ in range(num_inputs)]
        self.prev_bias= 0.0

        self.delta = 0.0

    #sigmoid activation function
    def sigmoid(self,x):
           if x >= 0:
              z = math.exp(-x)
              return 1 / (1 + z)
           else:
                z = math.exp(x)
                return z / (1 + z)
        # return 1.0/(1.0 + math.exp(-x))
    #sigmoid derivative that would be used in backpropagation
    def sigmoid_deriv(self,output):

        return output * (1 - output)
    #computing neurons output for a given input vector ,they would be saved for backpropagation
    def forward(self,input_vector):
        net_sum = self.bias
        for i in range(len(input_vector)):
            net_sum += input_vector[i] * self.weights[i]

        self.l_output= self.sigmoid(net_sum)  

 
        self.l_input= list(input_vector)

        return self.l_output

class NeuralNetwork:
    def __init__(self,architecture,learning_rate,momentum):

        #storing our hyperparameters
        self.lr = learning_rate
        self.mm = momentum
        self.architecture = architecture
        self.layers =[] #empty list to hold all the neurons

        #building the layers
        # we will loop over the hidden layer and output layer 

        for layer_idx in range(1,len(architecture)):
            num_neurons = architecture[layer_idx] #number of neurons in the layer
            num_inputs = architecture[layer_idx -1] #number of inputs the neuron receives from the previous layer
        
            #initialize each neuron with the number of inputs it expects
            layer = []

            for _ in range(num_neurons):
                neuron = Neuron(num_inputs)
                layer.append(neuron)

            self.layers.append(layer) # adds the completed kayer to the network
     #feedforward 
    def forward(self,input_vector):
        current_inputs = input_vector

        for layer in self.layers:
            next_inputs = []

            for neuron in layer:
                neuron_output = neuron.forward(current_inputs)
                next_inputs.append(neuron_output)

            current_inputs = next_inputs

        return current_inputs    

   #backpropagation to optimise weights and bias
    def backprop(self,target_vector):
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            output_layer = (layer_idx == len(self.layers)-1)

            for neuron_idx,neuron in enumerate(layer):
                if output_layer:
                    #output layer delts
                    error = neuron.l_output - target_vector[neuron_idx] 
                    neuron.delta = error * neuron.sigmoid_deriv(neuron.l_output)

                else:
                    #hidden layer delta
                    error_contrib = 0.0
                    next_layer = self.layers[layer_idx + 1]
                    for next_neuron in next_layer:
                        error_contrib+= next_neuron.delta * next_neuron.weights[neuron_idx]
                    neuron.delta = error_contrib * neuron.sigmoid_deriv(neuron.l_output)
 #function to update weight and biases using gradient descent with momentum
    def wb_updates(self):
        for layer in self.layers:
            for neuron in layer:
               #updating weights

               for i in range(len(neuron.weights)):
                  grad = neuron.delta * neuron.l_input[i]
                  update = self.lr * grad + self.mm * neuron.prev_weights[i]
                  neuron.weights[i] -= update
                  neuron.prev_weights[i] = update
                  
               #updating bias
               bias_update = self.lr * neuron.delta + self.mm * neuron.prev_bias
               neuron.bias -= bias_update
               neuron.prev_bias = bias_update
#defining the loss function 
#we are using mse
def mse(target,output):
        return sum((t - o)**2 for t,o in zip(target,output))/len(target)

#training our model
def train_net(nn,train_set,val_set = None,epochs = 100):
    for epoch in range(1,epochs + 1):
        total_loss = 0.0

        for x,y in train_set:
            output = nn.forward(x)
            #backpropagation
            nn.backprop(y)

            nn.wb_updates()

            total_loss += mse(y,output)

        avg_loss = total_loss/len(train_set)

#validation loss
        if val_set:
            val_loss = 0.0
            for x_val,y_val in val_set:
                output_val = nn.forward(x_val)
                val_loss += mse(y_val,output_val)
            val_loss/= len(val_set)
            
            print(f"{epoch}|Train Loss:{avg_loss:.6f}|Val Loss{val_loss:.6f}")

        else:
            print(f"{epoch}|Train Loss: {avg_loss:.6f}")
#saving models's parameters , weights and bias that have been updated and min and max values
def save_params(nn, X_min, X_max, Y_min, Y_max, filename="splits.txt"):
    with open(filename, "w") as f:

        f.write("X_MIN " + " ".join(map(str,X_min)) + "\n")
        f.write("X_MAX " + " ".join(map(str,X_max)) + "\n")
        f.write("Y_MIN " + " ".join(map(str,Y_min)) + "\n")
        f.write("Y_MAX " + " ".join(map(str,Y_max)) + "\n")

        f.write("TOPOLOGY " + " ".join(map(str, nn.architecture)) + "\n")

        for li, layer in enumerate(nn.layers):
            for ni, neuron in enumerate(layer):
                f.write(f"L{li}N{ni}_W " + " ".join(map(str, neuron.weights)) + "\n")
                f.write(f"L{li}N{ni}_B {neuron.bias}\n")
#initializing our network with number of layers,learning rate and momentum

nn = NeuralNetwork([2,7,2],learning_rate=0.5,momentum=0.5)
train_net(nn,train_set,val_set,epochs=100)

save_params (nn,X_min,X_max,Y_min,Y_max,"lander_params.txt")
            
            

               
               
               
 

            
               
            

            
            




  
