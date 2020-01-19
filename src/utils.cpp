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

// ----------------------------------------------------------

bool compare(const s_val_pos& svp1, const s_val_pos& svp2){
    return svp1.val<svp2.val;
}

double min(const Matrix_d& mat, uint col){
    double min=mat.get(1,col);
    // uint index=0;
    for(uint i=0;i<mat.shape().n_row-1;++i){
        if(min>mat.get(i+1,col)-mat.get(i,col) && mat.get(i+1,col)-mat.get(i,col)>0){
            min=mat.get(i+1,col)-mat.get(i,col);
            // index=i;
        }
    }
    return min;
}

double global_min(const Matrix_d& mat){
    double local_min=mat.get(0,0);
    for(uint i=0;i<mat.shape().n_row;++i){
        for(uint j=0;j<mat.shape().n_col;++j){
            if(local_min>mat.get(i,j)){
                local_min=mat.get(i,j);
            }
        }
    }
    return local_min;
}

uint max_index(const Matrix_d& mat, uint col){
    double max=mat.get(0,col);
    uint index=0;
    for(uint i=1;i<mat.shape().n_row;++i){
        if(max<mat.get(i,col)){
            max=mat.get(i,col);
            index=i;
        }
    }
    return index;
}

double global_max(const Matrix_d& mat){
    double local_max=mat.get(max_index(mat, 0),0);
    for(uint i=1;i<mat.shape().n_col;++i){
        if(local_max<mat.get(max_index(mat, i),i)){
            local_max=mat.get(max_index(mat, i),i);
        }
    }
    return local_max;
}

Matrix_d add_2cols(const Matrix_d& mat, double coef){
    uint n_row=mat.shape().n_row;
    uint n_col=mat.shape().n_col-1;

    Matrix_d result(n_row, n_col);
    for(uint i=0;i<n_row;++i){
        result[i][0]=mat.get(i,0)+mat.get(i,1)*coef;
        for(uint j=1;j<n_col;++j){
            result[i][j]=mat.get(i,j)+mat.get(i,j+1)*coef;
        }
    }
    return result;
}

// --------------------------------------------------------------

std::vector<s_val_pos> col2svps(const Matrix_d& mat, uint col, uint beg, uint end){
    std::vector<s_val_pos> svps;
    for(uint i=beg;i<=end;++i){
        svps.push_back(s_val_pos{mat.get(i,col), i});
        // std::cout<<"col2spvs #"<<i<<": "<<mat.get(i,col)<<std::endl;
    }
    return svps;
}

uint diff_index(const Matrix_d& mat, uint col, uint row){
    double curr=mat.get(row,col);
    uint index=mat.shape().n_row-1;
    // std::cout<<" shh: "<<mat.shape().n_row<<'\n';
    for(uint i=row+1;i<mat.shape().n_row;++i){
        if(std::abs(curr-mat.get(i,col))>eps){
            index=i-1;
            break;
        }
    }
    return index;
}

void swap_row(Matrix_d& mat, uint r1, uint r2){
    double tmp;
    for(uint i=0;i<mat.shape().n_col;++i){
        tmp=mat[r1][i];
        mat[r1][i]=mat[r2][i];
        mat[r2][i]=tmp;
    }
}

void modify_matrix(const std::vector<s_val_pos>& svps, Matrix_d& mat, uint beg, uint end){
    Matrix_d sorted=mat;
    std::vector<double> tmp;
    for(uint i=0;i<=end-beg;++i){
        sorted[beg+i]=mat[svps[i].pos];
        // std::cout<<svps[i].pos<<" : \n";
    }
    mat=sorted;
}

void dc_sort(Matrix_d& mat, uint col, uint beg, uint end){
    // static uint itr=0;
    // std::cout<<"dcs #"<<itr++<<" >>> col: "<<col<<" | ["<<beg<<", "<<end<<"]\n";
    
    if(beg!=end){
        std::vector<s_val_pos> svps=col2svps(mat, col, beg, end);
        std::sort(svps.begin(), svps.end(), compare);
        modify_matrix(svps, mat, beg, end);
        // std::cout<<mat<<'\n';
    }

    for(uint r=0;beg<end && col<mat.shape().n_col;++r){
        uint tmp_end=diff_index(mat, col, beg);
        // std::cout<<" > current for iteration #"<<r<<"\n";
        // std::cout<<"current column: "<<col<<"\n";
        // std::cout<<"new beg: "<<beg<<"\n";
        // std::cout<<"new end: "<<tmp_end<<"\n";
        dc_sort(mat, col+1, beg, tmp_end);
        beg=tmp_end+1;
        // sleep(1);
    }
    // std::cout<<"Finished ::: dcs >>> col: "<<col<<" | ["<<beg<<", "<<end<<"]\n";
}

double coefficient(const Matrix_d& mat, uint col){
    // std::cout<<"Found min at "<<col-1<<": "<<min(mat, col-1)<<'\n';
    // std::cout<<"Found max at "<<col<<": "<<mat[max_index(mat, col)][col]<<'\n';
    return min(mat, col-1)/mat.get(max_index(mat, col),col);
}

std::vector<double> coefficients(const Matrix_d& mat){
    static std::vector<double> coefs(1, 1.);
    static uint index=1;

    // std::cout<<"Matrix shape: "<<mat.shape()<<'\n';
    // std::cout<<"Global min: "<<global_min(mat)<<'\n';
    // std::cout<<"Global max: "<<global_max(mat)<<'\n';
    // std::cout<<"Index: "<<index<<"\n\n";

    uint n_row=mat.shape().n_row;
    uint n_col=mat.shape().n_col;

    coefs.push_back(0.9*min(mat, 0)/mat.get(max_index(mat, 1),1));
    if(n_col>2){
        Matrix tmp=add_2cols(mat, coefs[index++]);
        return coefficients(tmp);
    }
    return coefs;
}