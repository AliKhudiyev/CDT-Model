#include "cdt.hpp"

#include <cmath>

CDT::CDT(const DataSet& dataset, unsigned n_input, bool dc_sort){
    set(dataset, n_input, dc_sort);
    m_function.m_optimization=Optimization{NO_FIT, HALF, 5};
}

CDT::~CDT(){}

void CDT::set(const DataSet& dataset, unsigned n_input, bool dc_sort){
    m_dataset=dataset;
    m_layers.resize(3);
    
    unsigned n_output=dataset.shape().n_col-n_input;
    m_info.bias=m_dataset.biased();
    if(dc_sort) m_dataset.dc_sort();

    std::vector<Matrix_d> matrices=m_dataset.compile(n_output);
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

void CDT::fit(){
    double bias=m_info.bias;
    Matrix_d mat=m_dataset.matrix();
    std::cout<<"Dataset bias: "<<bias<<'\n';

    // std::cout<<"dc sort:\n"<<m_input<<'\n'<<m_output<<'\n';

    std::vector<double> keys;
    for(unsigned i=0;i<m_input.shape().n_row;++i)
        keys.push_back(0.);
    
    m_layers[0].m_weights[0][0]=1.;
    for(unsigned i=0;i<m_input.shape().n_col-1;++i){
        double& weight=m_layers[0].m_weights[0][i+1];
        for(unsigned t=0;t<m_input.shape().n_row;++t){
            keys[t]+=m_layers[0].m_weights[0][i]*m_input[t][i];
        }
        for(unsigned j=0;j<m_input.shape().n_row-1;++j){
            if(keys[j]<keys[j+1] &&
               m_input[j][i+1]>m_input[j+1][i+1]){
                   double tmp;
                   tmp=(keys[j+1]-keys[j])/m_input[j][i+1];
                   if(weight>tmp) weight=tmp;
               }
        }
    }
    unsigned col=m_input.shape().n_col;
    for(unsigned i=0;i<m_input.shape().n_row;++i){
        keys[i]+=m_layers[0].m_weights[0][col-1]*m_input[i][col-1];
    }

    std::cout<<"Updated weights: ";
    for(unsigned i=0;i<m_layers[0].m_weights.shape().n_col;++i){
        std::cout<<m_layers[0].m_weights[0][i]<<"\t";
    }   std::cout<<'\n';

    // Normalizing keys
    // std::cout<<"Keys: ";
    // for(unsigned i=0;i<keys.size();++i){
    //     std::cout<<keys[i]<<"\t";
    // } std::cout<<'\n';

    double key_bias=0.;
    for(unsigned i=0;i<m_layers[0].m_weights[0].size();++i){
        // std::cout<<" |> +"<<m_layers[0].m_weights[0][i]<<" * "<<bias<<'\n';
        key_bias+=m_layers[0].m_weights[0][i]*bias;
    }
    for(unsigned i=0;i<keys.size();++i){
        keys[i]-=key_bias;
    }
    
    std::cout<<"Key bias: "<<key_bias<<'\n';
    // std::cout<<"Keys: ";
    // for(unsigned i=0;i<keys.size();++i){
    //     std::cout<<keys[i]<<"\t";
    // }   std::cout<<'\n';

    std::cout<<" > Initializing intervals...\n";
    
    std::vector<Interval> intervals;
    double beg=0, end, output=m_output[0][0];
    for(unsigned i=0;i<keys.size();++i){
        if(m_output[i][0]==output){
            end=i;
        }
        else{
            // std::cout<<" dbg int: "<<keys[beg]<<", "<<keys[end]<<", "<<output<<'\n';
            intervals.emplace_back(keys[beg], keys[end], output);
            
            beg=++end;
            output=m_output[beg][0];
        }

        if(i==keys.size()-1){
            // std::cout<<" dbg int: "<<keys[beg]<<", "<<keys[end]<<", "<<output<<'\n';
            intervals.emplace_back(keys[beg], keys[end], output);
        }
    }
    std::cout<<"Overfit rate: "<<intervals.size()/(double)m_dataset.shape().n_row<<'\n';
    m_function.initialize(intervals);
}

double CDT::predict(const std::vector<double>& inputs) const{
    // std::cout<<"Prediction for (";
    // for(unsigned i=0;i<inputs.size();++i){
    //     std::cout<<inputs[i]<<" ";
    // }
    // std::cout<<"): ";

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

void CDT::test(std::ostream& out, const DataSet& dataset) const{
    unsigned all_guesses=dataset.shape().n_row;
    unsigned right_guesses=0;

    std::vector<Matrix_d> matrices=dataset.compile();
    for(unsigned i=0;i<all_guesses;++i){
        double output=predict(matrices[0][i]);
        // std::cout<<" : "<<output<<" vs "<<matrices[1][i][0]<<'\n';
        if(ceil(output)-output<0.5) output=ceil(output);
        else output=floor(output);
        if(abs(output-matrices[1][i][0])<0.00001){
            // std::cout<<"+\n";
            ++right_guesses;
        }
    }

    out<<"Accuracy: "<<(double)right_guesses/all_guesses<<'\n';
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
    double key=0., output;
    
    for(unsigned i=0;i<inputs.size();++i){
        key+=m_layers[0].m_weights.get(0,i)*inputs[i];
    }
    // std::cout<<" : key: "<<key<<'\n';
    output=m_function.run(key);

    return output;
}

void CDT::adjust_weights(unsigned cindex){
    if(m_cdt_indices.size()<2) return;
    
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
                tmp_weights.push_back(tmp_weight);
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
        cdt_index=insert(i);
        adjust_weights(cdt_index);
    }
}

void CDT::adjust_activation(){
    ;
}

void CDT::optimize(const Optimization& optimization){
    m_function.m_optimization=optimization;
}

unsigned CDT::insert(unsigned index){
    if(!index){
        m_cdt_indices.push_back(index);
        return 0;
    }
    
    unsigned cdt_index=m_cdt_indices.size();
    const std::vector<double>& inputs=m_input[index];

    for(unsigned i=0;i<m_cdt_indices.size();++i){
        for(unsigned j=0;j<m_input.shape().n_col;++j){
            if(inputs[j]<m_input[m_cdt_indices[i]][j]){
                cdt_index=i;
                i=m_cdt_indices.size();
                break;
            }
            else if(inputs[j]>m_input[m_cdt_indices[i]][j])
                break;
        }
    }
    m_cdt_indices.insert(m_cdt_indices.begin()+cdt_index, index);

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