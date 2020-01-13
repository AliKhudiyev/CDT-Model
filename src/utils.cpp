#include "utils.hpp"

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