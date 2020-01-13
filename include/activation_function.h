
#pragma once

#include <vector>

struct func_param{
    double middle, length, amplitude;
    unsigned confidence;
};

double gate(const func_param& param, const double input);
