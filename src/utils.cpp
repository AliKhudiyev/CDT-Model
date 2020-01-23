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

double min(const std::vector<double>& vec, unsigned stop){
    double min=vec[0];
    for(unsigned i=1;i<vec.size()-stop;++i){
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
        if(result>min(mat[i], 1)) result=min(mat[i], 1);
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

// ----------------------------------------------------------

std::vector<std::pair<unsigned, double>> get_vps(const Matrix_d& mat, unsigned col, unsigned beg_row, unsigned end_row){
    std::vector<std::pair<unsigned, double>> vps;

    for(unsigned i=beg_row;i<mat.shape().n_row && i<end_row;++i){
        vps.emplace_back(i,mat.get(i,col));
    }

    return vps;
}
unsigned diff_index(const Matrix_d& mat, unsigned col, unsigned beg_row, unsigned end_row){
    unsigned index=beg_row+1;
    double value=mat.get(beg_row,col);

    for(;index<end_row;++index){
        if(mat.get(index,col)!=value) break;
    }

    return index;
}
void modify_vps(Matrix_d& mat, const std::vector<std::pair<unsigned, double>>& vps, unsigned beg_row){
    Matrix_d result=mat;
    for(unsigned i=0;i<vps.size();++i)
        result[beg_row+i]=mat[vps[i].first];
    mat=result;
}
void sort_column(Matrix_d& mat, unsigned col, unsigned beg_row, unsigned end_row){
    std::vector<std::pair<unsigned, double>> vps=get_vps(mat, col, beg_row, end_row);
    
    std::sort(vps.begin(), vps.end(), 
        [](const std::pair<unsigned, double>& vp1, const std::pair<unsigned, double>& vp2)->bool{
            return vp1.second<vp2.second;
        }
    );

    modify_vps(mat, vps, beg_row);
}