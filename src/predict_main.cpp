#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main() {
    NeuralNetwork myNet(new Momentum(0.01, 0.9), new BinaryCrossEntropy());

    myNet.addLayer(8, 4, new Tanh());     
    myNet.addLayer(4, 8, new Tanh());     
    myNet.addLayer(1, 4, new Sigmoid());  

    cout << "Loading trained weights...\n";
    myNet.loadModel("data/trained_classification_model.csv");

    vector<double> test_input = {0.8, 0.7, 0.6, 0.5};

    cout << "\nRunning prediction...\n";
    vector<double> prediction = myNet.predict(test_input);
    

    cout << "Input Data: (" << test_input[0] << ", " << test_input[1] << ", " 
         << test_input[2] << ", " << test_input[3] << ")\n";    
    cout << "Network Output (Probability): " << prediction[0] << "\n";

    int predicted_class = (prediction[0] >= 0.5) ? 1 : 0;
    cout << "Predicted Class: " << predicted_class << "\n";

    return 0;
}