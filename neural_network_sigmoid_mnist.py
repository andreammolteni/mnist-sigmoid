import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 784, 1)
x_test  = x_test.reshape(-1, 784, 1)

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y].reshape(-1, num_classes, 1)

y_train = one_hot(y_train, 10)
y_test  = one_hot(y_test, 10)

N = 3  # number of layers in the network
m = 15 # dimension of each mini-batch
eta = 0.1 # learning rate
w=[np.random.normal(size=(30,784)),np.random.normal(size=(10,30))]  # weights matrices
b=[np.random.normal(size=(30,1)),np.random.normal(size=(10,1))]     # bias vectors



def crea_batches(x,y,m):
    return [
        (x[i:i+m],y[i:i+m])
         for i in range(0,len(x) - len(x) % m,m)
    ]
    
def sigmoid(z):
    return 1/(1+np.exp(-z))

def der_sigmoide(z):
    return sigmoid(z)*(1-sigmoid(z))

def cross_entropy(y,a):
    a=np.squeeze(a)
    a = np.clip(a, 10**-12, 1-10**-12)
    error=-np.sum(y * np.log(a) + (1-y)*np.log(1-a))
    return error

def feedforward(A,W,B):
    Z = np.dot(A,W.transpose()) + np.dot(np.ones(shape=(A.shape[0],1)),B.T)
    A1 = sigmoid(Z)
    return A1,Z

def accuratezza(x,y):
    c=0
    for i in range(x.shape[0]):
        A=x[i,:]
        if np.argmax(A)==np.argmax(y[i,:]):
            c+=1
    p = (c*100)/x.shape[0]
    return p

E = 80 # number of epochs
e=np.zeros(shape=(E*(60000//m),1))
ac_test=np.zeros(shape=(E,1))
ac_train=np.zeros(shape=(E,1))

for j in range(E):  # Iterate on the epochs
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_batch=x_train[indices]
    y_batch=y_train[indices]
    
    batches = crea_batches(x_batch,y_batch,m)
    
    for i,(x_batch,y_batch) in enumerate(batches): # Iterate on each batch
        bs=len(x_batch)
        
        gradb3=np.zeros(shape=(bs,10))
        gradb2=np.zeros(shape=(bs,30))
        
        gradw1=np.zeros(shape=(30,784))
        gradw2=np.zeros(shape=(10,30))
        error=0
        
        x_batch = x_batch.reshape(bs,784)
        for k in range(N-1):
            if k==0:
                Act=x_batch
            Act,Z = feedforward(Act,w[k],b[k])
            if k==0:
                att=Act
                z0=Z
        
        error = (1/m)*cross_entropy(Act,y_batch)
        der = der_sigmoide(z0)
        delta3 = Act - y_batch.reshape(m,10) 
        delta2 = np.dot(w[1].transpose(),((delta3.reshape(15,10)).transpose()))*der.transpose()
        gradb3 += delta3   # m x 10
        gradb2 += delta2.transpose()
        gradw1 += np.dot(delta2,x_batch.reshape(m,784))
        gradw2 += np.dot(delta3.T,att.reshape(m,30))
        
        e[j*len(batches)+i]=error
        
        b[1] -= (eta/bs)*np.sum(gradb3.T,axis=1,keepdims=True)
        b[0] -= (eta/bs)*np.sum(gradb2.T,axis=1,keepdims=True)
        w[0] -=(eta/bs)*gradw1
        w[1] -=(eta/bs)*gradw2
        
    # At the end of each batch (after having corrected bias and weights), compute the accuracy
    # both on training data and on test data-
    # (thus we perform feedforward, but not backpropagation, on both)
     
    for k in range(N-1):
            if k==0:
                Act=np.squeeze(x_train)
            Act,Z = feedforward(Act,w[k],b[k])
            if k==0:
                att=Act
                z0=Z
    ac_train[j]=accuratezza(Act,y_train) 
    
    for k in range(N-1):
            if k==0:
                Act=np.squeeze(x_test)
            Act,Z = feedforward(Act,w[k],b[k])
            if k==0:
                att=Act
                z0=Z
   
    ac_test[j]= accuratezza(Act,y_test)
             
          
batches=np.arange(len(e))
fig, ax = plt.subplots()

def media_mobile(vettore, k):
    v = np.array(vettore, dtype=float)
    risultato = []
    for i in range(len(v)):
        inizio = max(0, i - k)
        risultato.append(v[inizio:i+1].mean())
    return np.array(risultato)

v = [1, 2, 3, 4, 5]

ax.plot(batches, media_mobile(e,600))
ax.set_xlabel('Batches')
ax.set_ylabel('Error')
plt.show()

epochs=np.arange(len(ac_test))
fig,ax=plt.subplots()
ax.plot(epochs, ac_test,color='red',label='test')
ax.plot(epochs, ac_train,color='green',label='training')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy on test and training data')
ax.legend()
plt.show()

print(ac_test)
print(ac_train)