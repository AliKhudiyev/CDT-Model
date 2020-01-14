
#pragma once

#include <vector>

enum PARAM_NAME{
    MID_POINT=0,
    LENGTH,
    AMPLITUDE,
    CONFIDENCE
};

struct Func_Param{
    double middle, length, amplitude;
    unsigned confidence;
};

class Activation{
    private:
    std::vector<Func_Param> m_params;

    public:
    Activation()=default;
    ~Activation();

    void add(const Func_Param& param);
    void set(unsigned index, const PARAM_NAME& param_name, double value);
    void set(const PARAM_NAME& param_name, double value);
    void remove(unsigned index);
    void clear();

    double run(double input) const;
};

double gate(const Func_Param& param, double input);
