
#pragma once

#include <vector>
#include <string>

#include "matrix.hpp"

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

#define eps 0.000001

struct s_val_pos{
    double val;
    uint pos;
};

std::vector<s_val_pos> col2svps(const Matrix_d& mat, uint col, uint beg, uint end);
uint diff_index(const Matrix_d& mat, uint col, uint row);
void swap_row(Matrix_d& mat, uint r1, uint r2);
void modify_matrix(const std::vector<s_val_pos>& svps, Matrix_d& mat, uint beg, uint end);
// Data Cluster Sort
void dc_sort(Matrix_d& mat, uint col, uint beg, uint end);
double coefficient(const Matrix_d& mat, uint col);
std::vector<double> coefficients(const Matrix_d& mat);
