#include "layer.hpp"

Layer::Layer(unsigned n_perceptron){
    m_shape.n_row=0;
    m_shape.n_col=n_perceptron;
}

Layer::~Layer(){}

void Layer::add_perceptron(unsigned n_perceptron){
    m_shape.n_col+=n_perceptron;
}

void Layer::remove_perceptron(unsigned n_perceptron){
    if(m_shape.n_col<=n_perceptron){
        std::cout<<"ERROR [removing perceptron]: There has to be at least 1 perceptron!\n";
        exit(1);
    }
    m_shape.n_col-=n_perceptron;
}

void Layer::compile(unsigned row){
    m_shape.n_row=row;

    m_inputs.reshape(1, m_shape.n_col);
    m_outputs.reshape(1, row);
    m_weights.reshape(m_shape);

    m_weights.initialize(-0.1, 0.1);
}