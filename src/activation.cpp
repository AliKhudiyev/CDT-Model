#include "activation.hpp"

#include <cmath>
#include <iostream>

Activation::~Activation(){}

void Activation::initialize(const std::vector<Interval>& intervals){
    Func_Param param;
    double left_diff, right_diff;

    for(unsigned i=0;i<intervals.size()-1;++i){
        // std::cout<<" > Interval: "<<intervals[i].beg_key<<", "<<intervals[i].end_key<<" - "<<intervals[i].output<<'\n';
        if(intervals[i].output==0) continue;

        param.amplitude=intervals[i].output;
        param.confidence=m_optimization.confidence;

        if(i==0){
            left_diff=right_diff=(intervals[i+1].beg_key-intervals[i].end_key)/(double)(m_optimization.split_dist);
        }
        else if(i<intervals.size()-1){
            left_diff=(intervals[i].beg_key-intervals[i-1].end_key)/(double)m_optimization.split_dist;
            right_diff=(intervals[i+1].beg_key-intervals[i].end_key)/(double)m_optimization.split_dist;
        }
        else{
            left_diff=right_diff=(intervals[i].beg_key-intervals[i-1].end_key)/(double)(m_optimization.split_dist);
        }
        param.middle=(intervals[i].beg_key-left_diff+intervals[i].end_key+right_diff)/2.;
        param.length=param.middle-(intervals[i].beg_key-left_diff);
    
        // std::cout<<" dbg left/right diff: "<<left_diff<<"/"<<right_diff<<'\n';
        // std::cout<<" > param: "<<param.middle<<", "<<param.length<<", "<<param.amplitude<<", "<<param.confidence<<'\n';
        m_params.push_back(param);
    }
}

void Activation::add(const Func_Param& param){
    m_params.push_back(param);
}

void Activation::set(unsigned index, const PARAM_NAME& param_name, double value){
    if(index>=m_params.size()){
        std::cout<<"ERROR [activation set]: Index out of bounds!\n";
        exit(1);
    }
    
    switch (param_name)
    {
    case MID_POINT:
        m_params[index].middle=value;
        break;
    case LENGTH:
        m_params[index].length=value;
        break;
    case AMPLITUDE:
        m_params[index].amplitude=value;
        break;
    case CONFIDENCE:
        m_params[index].confidence=(unsigned)value;
        break;
    default: break;
    }
}

void Activation::set(const PARAM_NAME& param_name, double value){
    for(unsigned i=0;i<m_params.size();++i){
        set(i, param_name, value);
    }
}

void Activation::remove(unsigned index){
    if(index>=m_params.size()){
        std::cout<<"ERROR [activation removal]: Index out of bounds!\n";
        exit(1);
    }

    m_params.erase(m_params.begin()+index);
}

void Activation::clear(){
    m_params.clear();
}

double Activation::run(double input) const{
    double result=0.;
    for(const auto& param: m_params)
        result+=Activation::gate(param, input);
    return result;
}

double Activation::gate(const Func_Param& param, double input){
    return param.amplitude/(pow((input-param.middle)/param.length, 2*param.confidence)+1);
}