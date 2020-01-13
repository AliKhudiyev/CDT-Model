#include "activation_function.h"
#include <cmath>

double gate(const func_param& param, const double input){
    return param.amplitude/(pow(input-param.middle, 2*param.confidence)/param.length+1);
}