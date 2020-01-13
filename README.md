# CDT-Model

**Clustered Data Tree - Model** is an artificial neural network model which consists of only one hidden layer with a single perceptron.
Although common artificial neural networks with a single perceptron are not able to learn the data which are not linearly seperable
such as _XOr problem_ this model can learn any type of data. The way it does so, is to adjust activation functions of output nodes and dynamically update weights according some mathematical formula.

## Example
You can see how easy it is to create the model and learn the _XOr dataset_.
```
#include <iostream>
#include "cdt.hpp"

using namespace std;

int main(int argc, char* argv[]){

    DataSet xor("xor.csv");
    vector<DataSet> datasets=xor.split({70,30});

    CDT model;

    model.train(datasets[0], 2);
    model.test(datasets[1])<<endl;

    return 0;
}
```