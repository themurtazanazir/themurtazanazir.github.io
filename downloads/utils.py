import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def init_weights(self,layer_sizes,random_state):
        #save weights in a list of matrices
        np.random.seed(random_state)
        self.weights = [np.random.rand(layer_sizes[l-1]+1,layer_sizes[l])*(2/np.sqrt(layer_sizes[l-1]))-(1/np.sqrt(layer_sizes[l-1])) for l in range(1,len(layer_sizes))]
    def sigmoid(self,x):
        return 1/(1+np.exp(-x)) # keep beta = 1
    
    def forward(self,A_0,is_regression,weights=None):
        self.outputs=[]
        A_l = A_0
        self.outputs.append(A_0)
        if weights is None:
            weights = self.weights
        for weight in weights:
            A_lbias = np.concatenate(((-np.ones((A_l.shape[0],1)),A_l)),axis=1) # add bias to input data
            H_l = np.matmul(A_lbias,weight) # compute the summation
            A_l = self.sigmoid(H_l) # compute the activation
            self.outputs.append(A_l)
        if is_regression:
            A_l = H_l
            self.outputs.pop()
            self.outputs.append(A_l)
        return A_l # return the final output
            
    def backward(self,T, learning_rate,is_regression,batch_size,momentum,delta_w,loss):
        A_L = self.outputs[-1]
        if is_regression:
            delta_L = A_L-T
        elif loss == 'sumofsquares':
            delta_L = (A_L-T)*(A_L*(1-A_L)) # beta = 0
        elif loss == 'crossentropy':
#             delta_L = (T*(A_L-1)*A_L)
            delta_L= A_L-T
        delta_l_next = delta_L
        
        for i in range(len(self.weights)-1,-1,-1):
#             print(i)
            A_l = self.outputs[i]
            #compute error for previous layer
            delta_l = A_l*(1-A_l)*(np.matmul(delta_l_next,np.transpose(self.weights[i][1:,:])))
#             A_0 A_1 A_2
#             W_1 W_2
#             0   1    2
            # add bias output to output matrix
            A_lbias = np.concatenate(((-np.ones((A_l.shape[0],1)),A_l)),axis=1)
        
            # compute delta_w
            delta_w[i] = (1-momentum)*(1/batch_size)*(np.matmul(np.transpose(A_lbias),delta_l_next)) + momentum*delta_w[i]
            
            
            #update weights using the next errors
            self.weights[i] = self.weights[i] - learning_rate*delta_w[i]
            # change the next errors for next layer
            delta_l_next = delta_l
        return delta_w
            
    def compute_error(self,A_L,T,loss,epsilon=1e-100):
        if loss == 'sumofsquares':
            return np.sum((A_L-T)**2)/T.shape[0]
        
        if loss == 'crossentropy':
            return -np.sum(T*(np.log(A_L+epsilon))+(1-T)*(np.log(1-A_L+epsilon)))/T.shape[0]
            
            
            
    
    def train(self, input_data, input_target, epochs, layer_sizes=(100,), 
              learning_rate=0.01, batch_size=None,loss='sumofsquares', is_regression=False,
              momentum=0.9, random_state=0,verbose=0, save_weights=False,warm_start=False):
        
        
        X = np.array(input_data)
        Target = np.array(input_target)
        layer_sizes=list(layer_sizes)
        layer_sizes.insert(0,X.shape[1])
        n_outputs = Target.shape[1]
        layer_sizes.append(n_outputs)
        
        if not warm_start:
            self.init_weights(layer_sizes, random_state=random_state)
        if save_weights:
            self.saved_weights = [self.weights.copy()]
            
        if batch_size is None:
            batch_size=X.shape[0]
        
        # initialize delta_w to be zero for every layer
        delta_w = [0*i for i in self.weights].copy()
        for e in range(epochs):
            
                
            
    
            # shuffle the input so we don't train on same sequences
            idx = np.arange(0,Target.shape[0])
            np.random.shuffle(idx)
            X=X[idx]
            Target=Target[idx]
            
            b=0
            while b<X.shape[0]:
                A_0=X[b:b+batch_size,:]
                T=Target[b:b+batch_size,:]
                A_L = self.forward(A_0,is_regression)
                if e%((epochs//10)+1) == 0 and verbose:
                    print("epoch:",e)
                    print(f"Error: {self.compute_error(A_L,T,loss)}")
                delta_w = self.backward(T,learning_rate,is_regression,batch_size,momentum,delta_w,loss)
                

                if save_weights:
                    self.saved_weights.append(self.weights.copy())
                b=b+batch_size
        return self.compute_error(A_L,T,loss)
    
    def early_stopping(self,train,traintarget,val,valtarget,epochs,layer_sizes,learning_rate,is_regression,momentum,
                      diff=0.01,random_state=0,loss='sumofsquares'):
        error = self.train(train,traintarget,epochs,layer_sizes=layer_sizes,learning_rate=learning_rate,
                           is_regression=is_regression,momentum=momentum,loss=loss,random_state=random_state)
        old_val_error1 = 2000000000
        old_val_error2 = 1999999999
        
        new_error = self.compute_error(self.forward(val,is_regression=is_regression),valtarget,loss)
        
        
        count = 0
        while (((old_val_error1-new_error)>diff) or ((old_val_error2-old_val_error1)>diff)):
            count+=epochs
            error = self.train(train,traintarget,epochs,layer_sizes=layer_sizes,learning_rate=learning_rate,
                is_regression=is_regression,momentum=momentum,random_state=random_state,loss=loss,warm_start=True)
            
            

            old_val_error2=old_val_error1
            old_val_error1=new_error
            new_error = self.compute_error(self.forward(val,is_regression=is_regression),valtarget,loss)
        return new_error
    
    
    def predict(self,input_data,weights=None):
        output = self.forward(np.array(input_data),weights)
        #since this output is a realnumber(between 0 & 1)
        # we will have a threshold to predict its class for now 0.5
        if self.weights[-1].shape[1] is not 0:
            output = np.argmax(output,axis=1)
        else:
            output = (output>0.5)*1
            
        return output
    
    def confmat(self,input_data,targets):
        '''returns the confusion matrix for binary classification'''
        out = np.argmax(self.forward(input_data,is_regression=False),axis=1)
        t = np.argmax(targets,axis=1)
        mat = np.zeros([np.unique(t).shape[0]]*2)
        for i in np.unique(t):
            for j in np.unique(t):
                mat[i,j]=np.sum((out==i) & (t==j))
        return mat,np.sum(mat*np.eye(mat.shape[0]))/np.sum(mat)
    
    
    
def plot_decision_boundary(m,X,T,ax=False,weights=None):
    xx,yy=np.meshgrid(np.arange(X[:,0].min()-(X[:,0].max()-X[:,0].min()+1)/10,X[:,0].max()+(X[:,0].max()-X[:,0].min()+1)/10,(X[:,0].max()-X[:,0].min())/500),np.arange(X[:,1].min()-(X[:,1].max()-X[:,1].min()+1)/10,X[:,1].max()+(X[:,1].max()-X[:,1].min()+1)/10,(X[:,1].max()-X[:,1].min())/500))
    Z = m.predict(np.c_[xx.ravel(),yy.ravel()],weights)
    Z = Z.reshape(xx.shape)
    if not ax:
        ax = plt.figure().gca()
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax.set_ylabel('Input 2')
    ax.set_xlabel('Input 1')
    ax.scatter(X[:,0], X[:,1], c=T.squeeze(), cmap=plt.cm.Spectral)
    

    
# basic perceptron
import numpy as np

def compute_activations(X,W):
    activation = (np.matmul(X,W) > 0)
    return activation

def initialize_weights(n_input,n_out, random_state):
    np.random.seed(random_state)
    # the input shape should be including the bias inputs
    weights = np.random.rand(n_input, n_out)*0.1 - 0.05
    return weights

def update_weights(weights, input_matrix, output_matrix, target_matrix, learning_rate):
    delta_w = np.matmul(np.transpose(input_matrix),(output_matrix-target_matrix))
    weights = weights - (learning_rate*delta_w) # elementwise multiplication using broadcasting
    return weights

def train(input_data, target, learning_rate, epochs,random_state=0,init_weights=None, save_weights=False, verbose=False):
    # add the bias values to input_matrix
    X = np.concatenate((-np.ones((input_data.shape[0],1)),input_data),axis=1)
    #set the shapes
    n_input = X.shape[1]
    n_out = target.shape[1]
    
    #initialize the weights
    if init_weights is None:
        W = initialize_weights(n_input,n_out, random_state)
    else:
        W = init_weights
        
    if save_weights:
        weight_array=[W]
        
    for it in range(epochs):
        # compute outputs
        Y = compute_activations(X,W)
        
        if verbose:
            #print the output
            print(f"Iteration: {it}\n{W}\nOutput:\n{Y[:10,:10]}\nAccuracy: {(Y==target).sum()/X.shape[0]}")
        
        # update weights
        W = update_weights(W, X, Y, target, learning_rate)
        
        if save_weights:
            weight_array.append(W)
    
    print(f"Accuracy: {(Y==target).sum()/X.shape[0]}")    
    if save_weights:
        return W, weight_array
    else:
        return W
        
def recall(input_data, weights):
    # add the bias values to input_matrix
    X = np.concatenate((-np.ones((input_data.shape[0],1)),input_data),axis=1)
    Y = compute_activations(X,weights)
    return Y
    
    
