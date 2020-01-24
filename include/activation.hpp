
#pragma once

#include <vector>

class CDT;

enum PARAM_NAME{
    MID_POINT=0,
    LENGTH,
    AMPLITUDE,
    CONFIDENCE
};

enum FIT_FUNC{
    NO_FIT=0,
    LINEAR_FIT,
    AUTO_FIT
};

typedef enum SPLIT_DIST{
    AUTO=0,
    NONE,
    HALF,
    THIRD,
    FOURTH
}SPLIT_BORDER;

struct Interval{
    double beg_key, end_key, output;

    Interval()=default;
    Interval(double begk, double endk, double out):
        beg_key(begk), end_key(endk), output(out) {}
};

struct Func_Param{
    double middle, length, amplitude;
    unsigned confidence;
};

struct Optimization{
    FIT_FUNC fit_func;
    SPLIT_DIST split_dist;
    unsigned confidence;
};
class Activation{
    friend class CDT;

    public: // -> make private
    std::vector<Func_Param> m_params;
    Optimization m_optimization;

    public:
    Activation()=default;
    ~Activation();

    void initialize(const std::vector<Interval>& intervals);
    void add(const Func_Param& param);
    void set(unsigned index, const PARAM_NAME& param_name, double value);
    void set(const PARAM_NAME& param_name, double value);
    void remove(unsigned index);
    void clear();

    double run(double input) const;

    static double gate(const Func_Param& param, double input);
};
