import numpy as np


class layer:
    def __init__(self,neurons,activation_function):
        self.neurons=neurons
        self.Z=[0 for i in range(neurons)]
        self.A=[0 for i in range(neurons)]
        self.activation_function=activation_function
    def activation(self):
        if(self.activation_function=="RELU"):
            self.A=[max(0,i) for i in self.Z]
        elif(self.activation_function=="SIGMOID"):
            self.A=[1/(1+np.exp(-i)) for i in self.Z]

class neural_network:
    def __init__(self):
        self.layers=[]
        self.weights=[]
        self.bias=[]
        self.derivatives=[]

    def add_layer(self,layer):
        self.layers.append(layer)
        if(len(self.layers)>1):
            self.weights.append(np.random.rand(layer.neurons,self.layers[len(self.layers)-2].neurons))
            print("random weights",self.weights[-1])
            self.bias.append(np.random.rand(layer.neurons,1))
            self.derivatives.append(np.zeros(shape=(layer.neurons,self.layers[len(self.layers)-2].neurons)))

    def forward_pass(self,input):
        if(len(input)!=self.layers[0].neurons):
            print("Cant proceed, aborting")
            pass
        for layer in range(len(self.layers)):
            for neuron_idx in range(self.layers[layer+1].neurons):
                z=np.dot(input,self.weights[layer][neuron_idx])+self.bias[layer][neuron_idx][0]
                self.layers[layer+1].Z[neuron_idx]=z

            # print(self.layers[layer+1].Z)
            self.layers[layer+1].activation()
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
                # print(i,j)
                loss+=(i-j)**2
            loss/=len(expected)
        print(loss)
        return loss
    
    def backpropogate(self,exp_len):
        #Compute last derivatives
        output_layer_weight_rows=self.weights[len(self.weights)-1].shape[0]
        # print(output_layer_weight_rows)
        output_layer_weight_columns=self.weights[len(self.weights)-1].shape[1]

        for i in range(output_layer_weight_rows):
            for j in range(output_layer_weight_columns):
                # print("BEFORE->",self.derivatives[len(self.derivatives)-1][i][j])
                self.derivatives[len(self.derivatives)-1][i][j]=(2/output_layer_weight_rows) * self.layers[len(self.layers)-1].A[i] * self.layers[len(self.layers)-1].A[i] * (1-self.layers[len(self.layers)-1].A[i]) * self.layers[len(self.layers)-2].A[j]
                # print("AFTER->",i,j,self.derivatives[len(self.derivatives)-1][i][j])

        #For hidden layers
        for layer_idx in range(len(self.derivatives)-2,-1,-1):
            rows=self.weights[layer_idx].shape[0]
            cols=self.weights[layer_idx].shape[1]

            # print("BEFORE",self.weights[layer_idx])
            for i in range(rows):
                for j in range(cols):
                    self.derivatives[layer_idx][i][j]=sum([d[i] for d in self.derivatives[layer_idx+1]])*self.layers[layer_idx+1].A[j]
                    # print("##->",self.layers[layer_idx].A,len(self.derivatives),self.derivatives[layer_idx][i][j])
            self.weights[layer_idx]-=(self.derivatives[layer_idx]*0.01)
            # print("After",self.weights[layer_idx])


    

            


n=neural_network()
l1=layer(5,"SIGMOID")
l2=layer(6,"SIGMOID")
l3=layer(7,"SIGMOID")
n.add_layer(l1)
n.add_layer(l2)
n.add_layer(l3)

# print(n.weights)
# n.forward_pass([1,2,3,4,5])
# n.compute_loss("MSE",[1,1,1,0,1,0,1],n.layers[len(n.layers)-1].A)
# print([i.shape for i in n.weights])
# n.backpropogate(5)

for i in range(100000):
    n.forward_pass([1,2,3,4,5])
    if(i%10000==0):
        print([1,0,1,0,1,0,1],n.layers[-1].A)
        print(n.compute_loss("MSE",[1,0,0,0,0,0,0],n.layers[-1].A))
    n.backpropogate(7)


# w1x1+w2x2........... only x1 is considered for