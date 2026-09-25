#laoding all the necessary datasets
import math
from typing import List

#the neuron class we  that represents a single neuron with weights, 
#bias and the activation function
class Neuron:
    def __init__(self,num_inputs:int):

        #initializing weights and bias
        self.weights: List[float] = [0.0] * num_inputs
        self.bias: float = 0.0
        
        #storing the output and input
        self.l_output:float = 0.0


    #sigmoid activation function
    def sigmoid(self, x: float) -> float:
         if x >= 0:
              z = math.exp(-x)
              return 1 / (1 + z)
         else:
            z = math.exp(x)
            return z / (1 + z)


        
    
    #computing output of the neuron
    def forward(self,input_vector: List[float]) -> float:
        net_sum = self.bias
        for i in range(len(input_vector)):
            net_sum += input_vector[i] * self.weights[i]
    #we will use sigmoid function  
     
        self.l_output = self.sigmoid(net_sum)


        return self.l_output

class NeuralNetwork:
    def __init__(self,architecture: List[int]):

        #storing our hyperparameters
        self.architecture = architecture
        self.layers: List[List[Neuron]] =[] #empty list to hold all the neurons

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

            self.layers.append(layer) # adds the completed layer to the network

    def forward(self,input_vector: List[float]) -> List[float]:
        current_inputs = input_vector

        for layer in self.layers:
            next_inputs = []

            for neuron in layer:
                neuron_output = neuron.forward(current_inputs)
                next_inputs.append(neuron_output)

            current_inputs = next_inputs

        return current_inputs    

class NeuralNetHolder:

    def __init__(self):
        #paramaters from the training of the network,

        self.X_min= [-795.7259158087129, 65.51603340068675]
        self.X_max= [767.2665292395791, 469.30533290245734]
        self.Y_min= [-1.7960510631321673, -3.437111374171363]
        self.Y_max= [7.999999999999988, 3.419775471631748]
        
        #network architecture 2,7,2
        self.architecture =[2,7,2]
        self.nn = NeuralNetwork(self.architecture)
         #weights from the hidden layer
        weights_h_layer =[[-0.2348358869150835, -0.7909889350905646],
            [8.722858328930853, 41.89196816287567],
            [-30.85265876325794, -55.964048161378976],
            [0.058923751751343706, -0.12259463706461578],
            [6.308568896233476, -13.764278430820372],
            [-0.2748264479705362, -0.7085797116940814],
            [-7.535688862668526, 29.384582217575144]

                          ]
        #hidden layes biases from the trained network
        
        bias_h_layer= [  0.8265705582604852,
            -0.3476628493969072,
            0.3517707061443738,
            -0.5026229017060071,
            -0.6088805695651924,
            -0.23966434356356695,
            -0.5661938921089943
]

       #pre_trained weights from the output layer
        weights_o_layer =[  [0.301253048730977, 7.528825157982848, 0.69638352027907, 0.042705279930126856, 0.06696584810343101, 0.15283508906920495, -0.12358238090950804],
            [-0.9545811297023745, -3.8494425427090504, -2.0991268249494683, 0.04961198203909647, 10.37257008750927, 0.2178202492483298, -7.71252393507624]

]
                         
        #pre_trained output layer biases
        bias_o_layer = [5.3613862557215874, -2.12119031905349]

     #assigning weights and biases to the hidden neurons
        for idx,neuron in enumerate(self.nn.layers[0]):
            neuron.weights = weights_h_layer[idx]
            neuron.bias = bias_h_layer[idx]
    #assigning weights and biases to the hidden neurons
        for idx,neuron in enumerate(self.nn.layers[1]):
            neuron.weights = weights_o_layer[idx]
            neuron.bias = bias_o_layer[idx]

 
#normalizing and denormalizing our features
#scaling our data to a specified range
    def normalize(self, value: float, idx: int, is_input: bool = True) -> float:
        min_v = self.X_min[idx] if is_input else self.Y_min[idx]
        max_v = self.X_max[idx] if is_input else self.Y_max[idx]
      #clipping value to min/max range
        if value < min_v: value = min_v
        if value > max_v: value = max_v
        if max_v -min_v ==0:
            return 0.0
        return (value - min_v)/(max_v - min_v)
    #convert the scaled output back to the original range
    def denormalize(self, scaled_value: float, idx: int) -> float:
         min_v = self.Y_min[idx]
         max_v = self.Y_max[idx]
         return scaled_value * (max_v - min_v) + min_v



#network prediction
    def predict(self,input_row):
        cleaned = []
        for x in input_row:
            try:
                cleaned.append(float(x))
            except:
                cleaned.append(0.0)
        if len(cleaned) != 2: #making sure that the input vector has exactly 2 features
            cleaned =[0.0,0.0]





      #normaize input features
        x_norm = []
        for idx, val in enumerate(cleaned):
            scaled = self.normalize(val, idx, is_input=True)
    # clip to [0,1] because sigmoid network expects only this range
            scaled = max(0.0, min(1.0, scaled))
            x_norm.append(scaled)


      #forward pass through the network
        nn_output = self.nn.forward(x_norm)

       
       #denormalize output
        y_out = [self.denormalize(nn_output[i], i) for i in range(len(nn_output))]

        for i in range(len(y_out)):
            y_out[i] = max(min(y_out[i], self.Y_max[i]), self.Y_min[i])



        return y_out
        

