
#pragma once

#include "cdt.hpp"

enum RESULT_TYPE{
    AS_NUMBER=0,
    AS_LABEL,
    AS_CORRECTION
};

void save_results(const CDT& model, const DataSet& dataset, const std::string& filepath, const RESULT_TYPE& type=AS_LABEL, unsigned n_output=1);
