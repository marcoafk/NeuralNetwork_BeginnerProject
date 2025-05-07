import numpy as np
import special_functions as sf


class NeuralNetwork():
    #--------------------------------------------------------------------------
    #_________________Hyperparameters initialization___________________________
    def __init__(self, input_size, inner_sizes, output_size):
        self.init_params(input_size, inner_sizes, output_size)
        
    def init_params(self, input_size, inner_sizes, output_size):
        inner_sizes = np.array(inner_sizes)
        self.neuron_nums = np.array([input_size, *inner_sizes, output_size])
        self.layers_num = np.shape(self.neuron_nums)[0]

        L = self.layers_num - 1

        self.Weights = [
            np.random.randn(self.neuron_nums[i+1], self.neuron_nums[i]) * np.sqrt(2/self.neuron_nums[i])
            for i in range(L)
        ]
        self.Biases = [
            np.random.randn(self.neuron_nums[i+1], 1) * np.sqrt(2/self.neuron_nums[i])
            for i in range(L)
        ]
        print("Weights initialized")
        print(self.Weights)
        print("Biases initialized")
        print(self.Biases)
        
    #_________________Hyperparameters initialization___________________________
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #_____________________Main steps for the gradient descent__________________
    def forward_propagation(self, X):
        L = self.layers_num - 1 # Last layer index
        raw_output = [ [] for i in range(L)] 
        activated_output = [ [] for i in range(L)] 

        #first layer takes in the input
        raw_output[0] = self.Weights[0].dot(X) + self.Biases[0]
        activated_output[0] = sf.ReLu(raw_output[0])

        for i in range(1, L-1):
            raw_output[i] = self.Weights[i].dot(activated_output[i-1]) + self.Biases[i]
            activated_output[i] = sf.ReLu(raw_output[i])

        # last layer uses softmax for
        raw_output[L-1] = self.Weights[L-1].dot(activated_output[L-2]) + self.Biases[L-1]
        activated_output[L-1] = sf.softmax(raw_output[L-1])

        return raw_output, activated_output
            
    def back_propagation(self, raw_output, activated_output, 
                         X, Y):
        one_hot_Y = sf.convert_to_one_hot(Y)
        m_batch = X.shape[1] # num. of samples in this batch
        #________________________         ____________________________________
        L = self.layers_num - 1
        delta = [ [] for i in range(L)] 
        d_Weights = [ [] for i in range(L)]
        d_Biases = [ [] for i in range(L)]

        # Last layer 
        delta[L-1] = activated_output[L-1] - one_hot_Y
        d_Weights[L-1] = 1 / m_batch * delta[L-1].dot(activated_output[L-2].T)
        d_Biases[L-1] = 1 / m_batch * np.sum(delta[L-1], axis=1, keepdims=True)

        # Inner layers
        for i in range(L-1, 1, -1):
            delta[i-1] = (self.Weights[i].T.dot(delta[i]) 
                          * sf.ReLu_derivative(raw_output[i-1]))
            d_Weights[i-1] = 1 / m_batch * delta[i-1].dot(activated_output[i-2].T) 
            d_Biases[i-1] = 1 / m_batch * np.sum(delta[i-1], axis=1, keepdims=True)

        # First layer
        delta[0] = (self.Weights[1].T.dot(delta[1])
                    * sf.ReLu_derivative(raw_output[0]))    
        d_Weights[0] = 1 / m_batch * delta[0].dot(X.T)
        d_Biases[0] = 1 / m_batch * np.sum(delta[0], axis=1, keepdims=True)

        return d_Weights, d_Biases
        #________________________________________________________________


    def update_params(self, d_Weights, d_Biases, learning_rate):
        L = self.layers_num - 1
        for i in range(L-1, -1, -1):
            self.Weights[i] -= learning_rate * d_Weights[i]
            self.Biases[i] -= learning_rate * d_Biases[i]
    #_____________________Main steps for the gradient descent_______________________________________
    #-------------------------------------------------------------------------------

    def get_predictions(self, activated_outputs):
        return np.argmax(activated_outputs[-1], 0)
        
    def make_predictions(self, X):
        _, activated_output = self.forward_propagation(X)
        predictions = self.get_predictions(activated_output)
        return predictions

    def get_accuracy(self, predictions, Y):
        #print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def full_test(self, X, Y):
        _, full_data_output = self.forward_propagation(X)
        print("test_output", full_data_output[-1])
        print("test_output shape", full_data_output[-1].shape)
        predictions = self.get_predictions(full_data_output)
        print("predictions=",predictions)
        train_acc = self.get_accuracy(predictions, Y)

        return train_acc 
    #--------------------------------------------------------------------------------------
    
    def create_mini_batches(self, X, Y, batch_size):
        n_training_examples = X.shape[1] #gets the number of training examples
        random_index = np.random.permutation(n_training_examples)

        X_shuffled = X[ : , random_index] #all rows, but only the columns(examples) by randon_index
        Y_shuffled = Y[random_index]

        mini_batches = []
        for k in range(0, n_training_examples, batch_size):
            X_batch = X_shuffled[ : , k:k+batch_size]
            Y_batch = Y_shuffled[k:k+batch_size]
            
            mini_batches.append((X_batch, Y_batch))
            
        return mini_batches   
        
    #---------------------------------------------------------------------------------------
        
    def gradient_descent(self, X, Y, learning_rate, iterations, X_val=None, 
                         Y_val=None, patience=20, batch_size=64, verbose=True):
        accuracy_list = [] #for plotting the learning curve
        val_accuracy_list = [] #for checking performance while training
        best_val_acc = 0 #best test accuracy
        self.best_params = None
        counter = 0

        for i in range(iterations):
            mini_batches = self.create_mini_batches(X, Y, batch_size)

            for X_batch, Y_batch in mini_batches:
                #The actual training. Forward -> Back -> update -> Forward...
                (raw_output, 
                 activated_output) = self.forward_propagation(X_batch)
                #gradient 
                (d_Weights,
                 d_Biases) = self.back_propagation(raw_output, 
                                                   activated_output, 
                                                   X_batch, 
                                                   Y_batch)
                #learning
                self.update_params(d_Weights, d_Biases, learning_rate)

            #-----------------------------------------------------------------
            if i % 1 == 0: #-------------testing while training---------------
                #----
                # X, Y, verbose, X_val, Y_val
                #----
                train_acc = self.full_test(X, Y)
                accuracy_list.append(train_acc)
                if verbose:
                    print(f"Training acc: {train_acc}")

                if X_val is not None and Y_val is not None:
                    val_acc = self.full_test(X_val, Y_val)
                    val_accuracy_list.append(val_acc)

                    if val_acc > best_val_acc: #block for monitoring accuracy
                        best_val_acc = val_acc
                        best_params = (self.Weights, self.Biases)
                        counter = 0
                    else: #training stops once counter reaches patience
                        counter += 1

                    if verbose: 
                        print(f"Ite {i} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                    if counter >= patience:
                        if verbose: print("Stopped improving")
                        break

        if best_params: #----------early stopping triggered----------------
            return *best_params, accuracy_list, val_accuracy_list

        return (self.Weights, self.Biases, 
                accuracy_list, val_accuracy_list)
