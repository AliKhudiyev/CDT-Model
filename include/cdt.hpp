
#pragma once

#include "layer.hpp"
#include "dataset.hpp"
#include "activation_function.h"

class CDT{
    private:
    DataSet m_dataset;
    std::vector<Layer> m_layers;
    Matrix_d m_input, m_output;

    public:
    CDT()=default;
    CDT(const DataSet& dataset, unsigned n_input=1);
    ~CDT();

    void set(const DataSet& dataset, unsigned n_input=1);
    void compile(unsigned n_layer=3);

    void train();
    void train(const DataSet& dataset);
    double predict(const std::vector<double>& inputs) const;
    std::vector<double> predict(const Matrix_d& input_matrix) const;
    std::string label(const std::vector<double>& inputs) const;
    std::vector<std::string> labels(const Matrix_d& input_matrix) const;

    void load(const std::string& filepath);
    void save(const std::string& filepath) const;

    friend std::ostream& operator<<(std::ostream& out, const CDT& model);

    private:
    // These indexes are the indexes of CDT table
    std::vector<unsigned> m_cdt_indexes;

    double feed_forward(const std::vector<double>& inputs) const;
    void adjust_weights();
    void adjust_activation_functions();

    // Proper insertion of a new input vector to the CDT table
    void insert(const std::vector<double>& inputs);
    
    std::ostream& print(std::ostream& out) const;
};

std::ostream& operator<<(std::ostream& out, const CDT& model);
