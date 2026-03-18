import numpy as np
import math


def relu(x):                    #ReLU function. which is max{0,n}
    return np.maximum(0, x)

def sigmoid(x):                 #standard sigmoid function
    return 1/(1+np.exp(-x))



data = np.genfromtxt(r"D:\MACHINE LEARNING\diabetes.csv", delimiter=",", skip_header=1) #takes in the data from the csv file

x = data[:, :8]              #stores 8 inputs in x
y = data[:, 8].reshape(-1,1) #stores the output of the inputs in y

W1 = np.random.randn(16,8) * 0.01  #generation of random weights and biases for layer 1
B1 = np.zeros((16,1))

W2 = np.random.randn(8,16) * 0.01  #generation of random weights and biases for layer 1
B2 = np.zeros((8,1))

W3 = np.random.randn(1,8) * 0.01   #generation of random weights and biases for layer 1
B3 = np.zeros((1,1))


epoch=5000                         #training loop
for epochh in range(epoch):
    totalloss=0
    for sample in range(len(x)):
        X=x[sample].reshape(-1,1)                #gives optimum shape to do matrix multiplication which is 1x8
        Y=y[sample]
        Z1=np.dot(W1,X)+B1                       #does matrix multiplication of input and weights from layer 1 and adds biases. (1x8)x(8x16)=1x16
        a1=relu(Z1)
        Z2=np.dot(W2,a1)+B2                      #does matrix multiplication of input and weights from layer 2 and adds biases. (1x16)x(16x8)=1x8
        a2=relu(Z2)
        Z3=np.dot(W3,a2)+B3                      #does matrix multiplication of input and weights from layer 3 and adds biases. (1x8)x(8x1)=(1x1)
        a3=sigmoid(Z3)                           # this applies sigmoid to the value obtained in the 1x1 matrix.


        loss = -(Y*np.log(a3) + (1-Y)*np.log(1-a3))         #cross entropic loss function.
        totalloss+=loss


        DZ3=a3-Y                                #derivative of 3rd layer. DZ3=[d(L)/d(a3)]*[d(a3)/d(z3)]
        lis=[]
        for i in range(len(a2)):                #we need this loop since DZ2=[d(L)/d(a3)]*[d(a3)/d(z3)]*[d(z3)/d(a2)]*[d(a2)/d(z2)]
            if Z2[i]>0:                         #or DZ1=DZ2*[d(z3)/d(a2)]*[d(a2)/d(z2)]. and since d(a2)/d(z2)=1 or 0 depending upon a2 is +ve or 0
                lis.append(1)
            else:
                lis.append(0)
        lis=np.array(lis)
        DZ2=(W3.T*DZ3)*lis.reshape(-1,1)        #computing DZ2 as mentioned above

        lis1=[]
        for i in range(len(a1)):                #Similarly because of relu we need to perform the task in this layer as well.
            if Z1[i]>0:
                lis1.append(1)
            else:
                lis1.append(0)

        lis1=np.array(lis1)

        DZ1=(W2.T@DZ2)*lis1.reshape(-1,1)       #DZ1=[d(L)/d(a3)]*[d(a3)/d(z3)]*[d(z3)/d(a2)]*[d(a2)/d(z2)]*[d(z2)/d(a1)]*[d(a1)/d(z1)]
                                                #DZ1=DZ2*[d(z2)/d(a1)]*[d(a1)/d(z1)]
        lr = 0.01

        W3 = W3 - lr * np.dot(DZ3, a2.T)  #weights update
        W2 = W2 - lr * np.dot(DZ2, a1.T)
        W1 = W1 - lr * np.dot(DZ1, X.T)

        B3 = B3 - lr * DZ3                #biases update
        B2 = B2 - lr * DZ2
        B1 = B1 - lr * DZ1


correct = 0

for sample in range(len(x)):                 #displays percentage accuracy by comparison using the weights computed above.

    X = x[sample].reshape(-1,1)
    Y = y[sample]

    Z1 = W1 @ X + B1
    a1 = relu(Z1)

    Z2 = W2 @ a1 + B2
    a2 = relu(Z2)

    Z3 = W3 @ a2 + B3
    a3 = sigmoid(Z3)

    if a3>=0.5:
        pred=1
    else:
        pred=0

    if pred == Y:
        correct += 1

accuracy = correct / len(x)

print("Accuracy:", accuracy)
        