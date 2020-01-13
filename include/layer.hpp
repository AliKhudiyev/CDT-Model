
#pragma once

#include "matrix.hpp"

class CDT;

class Layer{
    friend class CDT;

    private:
    Shape m_shape;
    Matrix_d m_inputs, m_outputs;
    Matrix_d m_weights;

    public:
    Layer(): 
        Layer(1) {}
    Layer(unsigned n_perceptron);
    ~Layer();

    void add_perceptron(unsigned n_perceptron=1);
    void remove_perceptron(unsigned n_perceptron=1);
    void compile(unsigned row=1);
};
