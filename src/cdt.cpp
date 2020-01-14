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
        m_layers[i].compile(m_layers[i+1].m_shape.n_col);
    }
}

void CDT::train(){
    std::cout<<"Training..\n";
    std::cout<<"Input shape: "<<m_input.shape()<<'\n';
    std::cout<<"Output shape: "<<m_output.shape()<<'\n';

    std::cout<<"Initial weights: ";
    for(unsigned i=0;i<m_layers[0].m_weights.shape().n_col;++i){
        std::cout<<m_layers[0].m_weights[0][i]<<" ";
    }   std::cout<<'\n';

    adjust_weights();
    adjust_activation();

    std::cout<<"Updated weights: ";
    for(unsigned i=0;i<m_layers[0].m_weights.shape().n_col;++i){
        std::cout<<m_layers[0].m_weights[0][i]<<" ";
    }   std::cout<<'\n';
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

void CDT::adjust_weights(unsigned cindex){
    if(m_cdt_indices.size()<2) return;
    
    double semi_keys[2];
    std::vector<unsigned> cdt_indices;

    // std::cout<<" dbg > cdt indices: ";
    // for(unsigned i=0;i<m_cdt_indices.size();++i){
    //     std::cout<<m_cdt_indices[i]<<" ";
    // }
    // std::cout<<'\n';

    cdt_indices.push_back(m_cdt_indices[cindex]);
    if(cindex==0){
        cdt_indices.push_back(m_cdt_indices[cindex+1]);
    }
    else if(cindex==m_cdt_indices.size()-1){
        cdt_indices.insert(cdt_indices.begin(), m_cdt_indices[cindex-1]);
    }
    else{
        cdt_indices.insert(cdt_indices.begin(), m_cdt_indices[cindex-1]);
        cdt_indices.push_back(m_cdt_indices[cindex+1]);
    }

    // std::cout<<" dbg cdt indices: ";
    // for(const auto& index: cdt_indices)
    //     std::cout<<index<<" ";
    // std::cout<<'\n';

    for(unsigned i=0;i<cdt_indices.size()-1;++i){
        std::vector<double> tmp_weights;
        tmp_weights.push_back(1.);
        for(unsigned j=0;j<m_input.shape().n_col-1;++j){
            if(m_input[cdt_indices[i]][j]!=m_input[cdt_indices[i+1]][j] &&
               m_input[cdt_indices[i]][j+1]>m_input[cdt_indices[i+1]][j+1]){
                double tmp_weight;
                tmp_weight=(m_input[cdt_indices[i+1]][j]-m_input[cdt_indices[i]][j]);
                tmp_weight/=m_input[cdt_indices[i]][j+1];
                tmp_weights.push_back(tmp_weight/2.0);
                // std::cout<<" dbg tmp weight: "<<tmp_weights[tmp_weights.size()-1]<<" ? "<<m_layers[0].m_weights[0][j+1]<<'\n';
            } else{
                tmp_weights.push_back(10.);
            }
        }
        // std::cout<<"processing on weights..\n";
        keep_least_weights(m_layers[0].m_weights[0], tmp_weights);
        // std::cout<<"kept the least weights!\n";
    }
    // std::cout<<" dbg weight matrix shape: "<<m_layers[0].m_weights.shape()<<'\n';
}

void CDT::adjust_weights(){
    unsigned cdt_index;
    for(unsigned i=0;i<m_input.shape().n_row;++i){
        // std::cout<<" : "<<m_input[i][0]<<", "<<m_input[i][1]<<'\n';
        cdt_index=insert(i);
        // std::cout<<" dbg cdt index: "<<cdt_index<<'\n';
        adjust_weights(cdt_index);
    }
}

void CDT::adjust_activation(){
    ;
}

unsigned CDT::insert(unsigned index){
    if(!index){
        // std::cout<<" dbg initial insertion with inserted index: 0\n";
        m_cdt_indices.push_back(index);
        return 0;
    }
    
    unsigned cdt_index=m_cdt_indices.size();
    const std::vector<double>& inputs=m_input[index];

    for(unsigned i=0;i<m_cdt_indices.size();++i){
        for(unsigned j=0;j<m_input.shape().n_col;++j){
            // std::cout<<" dbg ("<<i<<","<<j<<") versus: "<<inputs[j]<<" and "<<m_input[m_cdt_indices[i]][j]<<'\n';
            if(inputs[j]<m_input[m_cdt_indices[i]][j]){
                // std::cout<<" > versus accepted!\n";
                cdt_index=i;
                // i=m_input.shape().n_row;
                i=m_cdt_indices.size();
                break;
            }
            else if(inputs[j]>m_input[m_cdt_indices[i]][j])
                break;
        }
    }
    m_cdt_indices.insert(m_cdt_indices.begin()+cdt_index, index);
    // std::cout<<" dbg inserted index: "<<index<<'\n';
    return cdt_index;
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