#include "utils.hpp"
#include "cdt.hpp"

#include <iostream>

void save_results(const CDT& model, const DataSet& dataset, const std::string& filepath, const RESULT_TYPE& type, unsigned n_output){
    std::vector<Matrix_d> matrices=dataset.compile(n_output);
    std::ofstream file(filepath);

    if(!file){
        std::cout<<"ERROR [saving results]: Couldn't open up the file!\n";
        exit(1);
    }

    if(type==AS_NUMBER){
        std::vector<double> outputs=model.predict(matrices[0]);
        // TO DO
    } else{
        std::vector<std::string> labels=model.labels(matrices[0]);
        // TO DO
    }

    file.close();
}

void keep_least_weights(std::vector<double>& old_weights, const std::vector<double>& new_weights){
    // std::cout<<" dbg size comparison: "<<old_weights.size()<<" vs "<<new_weights.size()<<'\n';
    // std::cout<<" dbg new weights: ";
    for(unsigned i=0;i<old_weights.size();++i){
        // std::cout<<" \ndbg check if "<<old_weights[i]<<" > "<<new_weights[i]<<" ? => ";
        if(old_weights[i]>new_weights[i]){
            old_weights[i]=new_weights[i];
        }
        // std::cout<<old_weights[i]<<" ";
    }
    // std::cout<<'\n';
}

double min(const std::vector<double>& vec){
    double min=vec[0];
    for(unsigned i=1;i<vec.size();++i){
        if(min>vec[i]) min=vec[i];
    }
    return min;
}

double max(const std::vector<double>& vec){
    double max=vec[0];
    for(unsigned i=1;i<vec.size();++i){
        if(max<vec[i]) max=vec[i];
    }
    return max;
}

double min(const DataSet& dataset){
    Matrix_d mat=dataset.matrix();
    double result=mat[0][0];
    for(unsigned i=0;i<mat.shape().n_row;++i){
        if(result>min(mat[i])) result=min(mat[i]);
    }
    return result;
}

double max(const DataSet& dataset){
    Matrix_d mat=dataset.matrix();
    double result=mat[0][0];
    for(unsigned i=0;i<mat.shape().n_row;++i){
        if(result<max(mat[i])) result=max(mat[i]);
    }
    return result;
}