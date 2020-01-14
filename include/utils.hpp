
#pragma once

#include <vector>
#include <string>

class DataSet;
class CDT;

enum RESULT_TYPE{
    AS_NUMBER=0,
    AS_LABEL,
    AS_CORRECTION
};

void save_results(const CDT& model, const DataSet& dataset, const std::string& filepath, const RESULT_TYPE& type=AS_LABEL, unsigned n_output=1);
void keep_least_weights(std::vector<double>& old_weights, const std::vector<double>& new_weights);
double min(const std::vector<double>& vec);
double max(const std::vector<double>& vec);
double min(const DataSet& dataset);
double max(const DataSet& dataset);