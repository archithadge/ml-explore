import numpy as np


class layer:
    def __init__(self,neurons,activation_function):
        self.neurons=neurons
        self.A=[0 for i in range(neurons)]
        self.activation_function=activation_function
    def activation(self):
        if(self.activation_function=="RELU"):
            self.A=[max(0,i) for i in self.A]
        elif(self.activation_function=="SIGMOID"):
            self.A=[1/(1+np.exp(-i)) for i in self.A]

class neural_network:
    def __init__(self):
        self.layers=[]
        self.weights=[]
        self.bias=[]
    def add_layer(self,layer):
        self.layers.append(layer)
        if(len(self.layers)>1):
            self.weights.append(np.random.rand(layer.neurons,self.layers[len(self.layers)-2].neurons))
            self.bias.append(np.random.rand(layer.neurons,1))

    def forward_pass(self,input):
        if(len(input)!=self.layers[0].neurons):
            print("Cant proceed, aborting")
            pass
        for layer in range(len(self.layers)):
            for neuron_idx in range(self.layers[layer+1].neurons):
                # print(self.layers[layer+1].neurons,input,self.weights[layer][neuron_idx])
                # print(self.bias[layer][neuron_idx])
                a=np.dot(input,self.weights[layer][neuron_idx])+self.bias[layer][neuron_idx][0]
                # print(a)
                self.layers[layer+1].A[neuron_idx]=a

            # print(self.layers[layer+1].A)
            self.layers[layer+1].activation()
            # print(self.layers[layer+1].A)
            if(layer==len(self.layers)-2):
                break
            input=self.layers[layer+1].A

    def compute_loss(self,loss_function,expected,actual):
        if(len(actual)!=len(expected)):
            print("Can't proceed")
            pass
        if(loss_function=="MSE"):
            loss=0
            for i,j in zip(expected,actual):
                print(i,j)
                loss+=(i-j)**2
            loss/=len(expected)
        print(loss)
        return loss
    

            


n=neural_network()
l1=layer(5,"SIGMOID")
l2=layer(6,"SIGMOID")
l3=layer(7,"SIGMOID")
n.add_layer(l1)
n.add_layer(l2)
n.add_layer(l3)

# print(n.weights)
n.forward_pass([1,2,3,4,5])
n.compute_loss("MSE",[1,0,1,0,1,0,1],n.layers[len(n.layers)-1].A)




