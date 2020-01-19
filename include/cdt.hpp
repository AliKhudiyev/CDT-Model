
#pragma once

#include "layer.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "utils.hpp"

class CDT{
    private:
    DataSet m_dataset;
    std::vector<Layer> m_layers;
    Matrix_d m_input, m_output;
    Activation m_function;

    public:
    CDT()=default;
    CDT(const DataSet& dataset, unsigned n_input=1);
    ~CDT();

    void set(const DataSet& dataset, unsigned n_input=1);
    void compile(unsigned n_layer=3);

    void train();
    void train(const DataSet& dataset);
    void fit(const DataSet& dataset);
    double predict(const std::vector<double>& inputs) const;
    std::vector<double> predict(const Matrix_d& input_matrix) const;
    std::string label(const std::vector<double>& inputs) const;
    std::vector<std::string> labels(const Matrix_d& input_matrix) const;

    void load(const std::string& filepath);
    void save(const std::string& filepath) const;

    friend std::ostream& operator<<(std::ostream& out, const CDT& model);

    private:
    // These indexes are the indexes of CDT table
    std::vector<unsigned> m_cdt_indices;

    double feed_forward(const std::vector<double>& inputs) const;
    void adjust_weights(unsigned cindex);
    void adjust_weights();
    void adjust_activation();

    // Proper insertion of a new input vector to the CDT table
    unsigned insert(unsigned index);
    
    std::ostream& print(std::ostream& out) const;
};

std::ostream& operator<<(std::ostream& out, const CDT& model);
