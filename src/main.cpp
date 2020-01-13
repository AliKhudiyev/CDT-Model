#include <iostream>

#include "matrix.hpp"
#include "dataset.hpp"
#include "cdt.hpp"
#include "utils.hpp"

#define EXAMPLE(filepath)   (string("../examples/")+filepath)

using namespace std;

int main(){

    // ReadInfo info{true, true, true, true};

    // DataSet dataset("new.csv", READ(00011));
    // cout<<dataset.labels_string()<<'\n';
    // cout<<dataset<<'\n';

    // vector<DataSet> datasets=dataset.split({20,30,50});
    // for(auto& ds: datasets){
    //     ds.shuffle();
    //     cout<<'\n'<<ds;
    // }

    // datasets[1].save("new.csv", WRITE(000));

    DataSet dataset(EXAMPLE("example.csv"), READ(00011));

    CDT model(dataset, 2);
    model.compile();

    cout<<model;

    return 0;
}