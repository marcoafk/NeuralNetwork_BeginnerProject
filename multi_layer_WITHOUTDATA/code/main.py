import NeuralNetwork as NN
import data_handling as dataH
import matplotlib.pyplot as plt



if __name__ == "__main__":

    #-----------data stuff--------------------
    X_train, Y_train, X_test, Y_test = dataH.load_data()

    #--------------------------neural network initialization--------------------------------
    inner_layers = [30, 25, 20]
    neural_network = NN.NeuralNetwork(784, inner_layers, 10) #pixels of each image, hidden neurons, output neurons

    #----------------training parameters------------------------------------------------
    learning_rate = 0.01 
    iterations = 50
    results = neural_network.gradient_descent(X_train, Y_train, learning_rate, iterations, X_test, Y_test, batch_size=10)
    
    #---------------plotting performance---------------
    train_accuracy = results[2]
    test_accuracy = results[3]
    
    """
    #-------- plotting stuff-------  
    plt.plot(range(0, iterations), train_accuracy, label='training accuracy')
    plt.plot(range(0, iterations), test_accuracy, label='test accuracy')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy over Time")
    plt.grid(True)
    plt.legend()
    plt.show()
    #
    # BUGS WHEN EARLY STOPPING IS TRIGGERED
    """