#include "cdt.hpp"

CDT::CDT(const DataSet& dataset, unsigned n_input){
    set(dataset, n_input);
}

CDT::~CDT(){}

void CDT::set(const DataSet& dataset, unsigned n_input){
    m_dataset=dataset;
    m_layers.resize(3);
    
    unsigned n_output=dataset.shape().n_col-n_input;

    std::vector<Matrix_d> matrices=dataset.compile(n_output);
    m_input=matrices[0];
    m_output=matrices[1];
}

void CDT::compile(unsigned n_layer){
    m_layers.resize(n_layer);
    
    m_layers[0].add_perceptron(m_input.shape().n_col-1);
    for(unsigned i=0;i<m_layers.size()-1;++i){
        m_layers[i].compile(m_layers[i+1].m_shape.n_col-1);
    }
}

void CDT::train(){
    // TO DO
}

void CDT::train(const DataSet& dataset){
    set(dataset);
    train();
}

double CDT::predict(const std::vector<double>& inputs) const{
    return feed_forward(inputs);
}

std::vector<double> CDT::predict(const Matrix_d& input_matrix) const{
    std::vector<double> results;
    for(unsigned i=0;i<input_matrix.shape().n_row;++i)
        results.push_back(predict(input_matrix.get(i)));
    return results;
}

std::string CDT::label(const std::vector<double>& inputs) const{
    std::string label;
    // TO DO
    return label;
}

std::vector<std::string> CDT::labels(const Matrix_d& input_matrix) const{
    std::vector<std::string> labels;
    for(unsigned i=0;i<input_matrix.shape().n_row;++i)
        labels.push_back(label(input_matrix.get(i)));
    return labels;
}

void CDT::load(const std::string& filepath){
    std::ifstream file(filepath);

    if(!file){
        std::cout<<"ERROR [loading model]: Couldn't open up the file!\n";
        exit(1);
    }

    // TO DO

    file.close();
}

void CDT::save(const std::string& filepath) const{
    std::ofstream file(filepath);

    if(!file){
        std::cout<<"ERROR [saving model]: Couldn't open up the file!\n";
        exit(1);
    }

    // TO DO

    file.close();
}

double CDT::feed_forward(const std::vector<double>& inputs) const{
    double output;
    ;
    return output;
}

void CDT::adjust_weights(){
    ;
}

void CDT::adjust_activation_functions(){
    ;
}

std::ostream& CDT::print(std::ostream& out) const{
    out<<"\tCDT-Model structure.\n";
    
    out<<" | Data set status: ";
    if(!m_dataset.empty()){
        out<<"Ready.\n";

        out<<" | Feautures: ";
        std::vector<std::string> labels=m_dataset.labels();
        for(unsigned i=0;i<labels.size();++i){
            out<<labels[i];
            if(i<labels.size()-1) out<<", ";
        }
        out<<'\n';
    } else  out<<"None.\n";

    out<<" | # of layers: "<<m_layers.size()<<"\n";
    out<<" \t";
    for(const auto& layer: m_layers)
        out<<"|- "<<layer.m_shape.n_col<<" -|";
    out<<'\n';

    out<<" | Weight structure:";
    out<<'\n';

    return out;
}

std::ostream& operator<<(std::ostream& out, const CDT& model){
    return model.print(out);
}