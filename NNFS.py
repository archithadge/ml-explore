import numpy as np
from nnfs.datasets import sine_data,spiral_data,vertical_data



def sigmoid(x,derivative=False):
    result=1/(1+np.exp(-x))
    if(derivative):
        return result*(1-result)
    return result

def relu(x):
    return max(0,x)

class layer:
    def __init__(self,neurons):
        self.input=[0 for i in range(neurons)]
        self.output=[0 for i in range(neurons)]
        self.neurons=neurons
        self.input_derivatives=[0 for i in range(neurons)]
        self.biases=np.random.rand(neurons,1)
        self.bias_derivatives=[0 for i in range(neurons)]

    def set_input(self,inputs):
        self.input=inputs

    def set_output(self,output):
        self.output=output

    def set_input_derivatives(self,input_derivatives):
        self.input_derivatives=input_derivatives

    def set_biases(self,biases):
        self.biases=biases

    def set_bias_derivatives(self,bias_derivatives):
        self.bias_derivatives=bias_derivatives

    def get_input(self):
        return self.inputs

    def get_output(self):
        return self.output

    def get_input_derivatives(self):
        return self.input_derivatives

    def get_biases(self):
        return self.biases

    def get_bias_derivatives(self):
        return self.bias_derivatives
    
    def activate(self,activation_function="SIGMOID"):
        if(activation_function=="SIGMOID"):
            self.set_output([sigmoid(i) for i in self.input])
    

class neural_network:
    def __init__(self):
        self.weights=[]
        self.weight_derivatives=[]
        self.layers=[]

    def print_layers(self):
        for i in range(len(self.layers)):
            print("Layer ",i)
            print("Neurons->",self.layers[i].neurons)
            print("Inputs",self.layers[i].input)
            print("Outputs",self.layers[i].output)
            print("*****************")



    def add_layer(self,layer):
        self.layers.append(layer)
        if(len(self.layers)>1):
            l1=self.layers[-1].neurons
            l2=self.layers[-2].neurons
            self.weights.append(np.random.rand(l1,l2))
            self.weight_derivatives.append(np.random.rand(l1,l2))

    def forward_propogation(self,input):
        input_neurons=self.layers[0].neurons
        if(len(input)!=input_neurons):
            print("Size of input isn't equal to neurons in input layer, aborting")
            pass

        self.layers[0].set_output(input)

        for idx in range(1,len(self.layers)):
            for j in range(self.layers[idx].neurons):
                self.layers[idx].input[j]=np.sum(input*self.weights[idx-1][j])+self.layers[idx].biases[j]
            self.layers[idx].activate()
            input=self.layers[idx].output

    def compute_input_derivatives(self,output):
        #For last layer
        for i in range(self.layers[-1].neurons):
            self.layers[-1].input_derivatives[i]=(-2/self.layers[-1].neurons)*(output[i]-self.layers[-1].output[i])*(self.layers[-1].output[i])*(1-self.layers[-1].output[i])
        #For rest of the layers
        for layer_idx in range(len(self.layers)-2,-1,-1):
            for j in range(self.layers[layer_idx].neurons):
                self.layers[layer_idx].input_derivatives[j]=np.sum(self.layers[layer_idx+1].input_derivatives*np.reshape([self.weights[layer_idx][k][j] for k in range(self.layers[layer_idx+1].neurons)],(self.layers[layer_idx+1].neurons,1)))*self.layers[layer_idx].output[j]*(1-self.layers[layer_idx].output[j])
                

        
        
    def compute_weight_derivatives(self,output):
        #for last layer
        rows,cols=self.weights[-1].shape[0],self.weights[-1].shape[1]
        for row in range(rows):
            for col in range(cols):
                self.weight_derivatives[-1][row][col]=(-2/rows)*(output[row]-self.layers[-1].output[row])*(self.layers[-1].output[row])*(1-self.layers[-1].output[row])*self.layers[-2].output[col]

        

        #for hidden layers
        for layer_idx in range(len(self.weights)-2,-1,-1):
            rows,cols=self.weights[layer_idx].shape[0],self.weights[layer_idx].shape[1]
        
            for row in range(rows):
                for col in range(cols):
                    self.weight_derivatives[layer_idx][row][col]=self.layers[layer_idx+1].input_derivatives[row]*self.layers[layer_idx].output[col]

 

            



        pass
    def compute_bias_derivatives(self):
        for i in range(len(self.layers)):
            self.layers[i].bias_derivatives=self.layers[i].input_derivatives
        pass

    def compute_accuracy(self,actual,predicted):
        correct=0
        for i in range(len(actual)):
            if(predicted[i]==actual[i]):
                correct+=1
        print("Accuracy is ",(correct/len(actual))*100)
        return (correct/len(actual))*100

    def compute_loss(self,output):
        sum=0
        for i in range(len(output)):
            sum+=(output[i]-self.layers[-1].output[i])**2
        loss=sum/len(output)
        print("Loss is ",sum/len(output))



    def train(self,epoch,trainX,trainY,testX,testY,learning_rate=0.01):
        if(len(trainX)!=len(trainY) or len(testX)!=len(testY)):
            print("Dimension mismatch")
        
        predicted_outputs=[]
        accuracies=[]
        for ep in range(epoch):
            print("Epoch=>",ep,"LR->",learning_rate,end=' ',)
            predicted_outputs=[]
            for i in range(len(trainX)):
                # print(i)
                self.forward_propogation(trainX[i])
                self.compute_input_derivatives(trainY[i])
                self.compute_weight_derivatives(trainY[i])
                self.compute_bias_derivatives()
                for k in range(len(self.weight_derivatives)):
                    self.weights[k]-=self.weight_derivatives[k]*learning_rate

                for j in range(len(self.layers)):
                    for k in range(len(self.layers[j].biases)):
                        self.layers[j].biases[k]-=self.layers[j].bias_derivatives[k]*learning_rate

                predicted_output=[1 if max(self.layers[-1].output)==j else 0 for j in self.layers[-1].output]
                # predicted_output=self.layers[-1].output[0][0]
                predicted_outputs.append(predicted_output)
                # print(predicted_output,trainY[i])
                # self.compute_loss(trainY[i])
            accuracies.append(self.compute_accuracy(trainY,predicted_outputs))
            # print(accuracies[-11,-1])
            # if(len(accuracies)>=11 and max(accuracies)>=0.8 and len(set(accuracies[-11:-1]))==1):
            #     learning_rate=learning_rate/2
            #     print("Learning rate changed")
            # print(predicted_outputs)


def onehot(val,classes):
    l=[0 for i in range(classes)]
    l[val]=1
    return l           

samples=5000
classes=10
data=vertical_data(samples,classes)
# print(data)

n=neural_network()
neurons=[2,4,5,4,classes]
for i in neurons:
    n.add_layer(layer(i))


trainX=data[0]
d={i:onehot(i,classes) for i in range(classes)}
# trainY=np.reshape(data[1],(len(data[1]),1))
trainY=[d[i] for i in data[1]]

n.train(10000,trainX,trainY,[],[],0.5)


