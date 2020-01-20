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

    DataSet dataset(EXAMPLE("xor.csv"), READ(01011));
    dataset.shuffle();
    cout<<dataset<<'\n';

    CDT model(dataset, 2);
    model.compile();
    // model.train();
    model.fit(dataset);

    cout<<"\n > Testing stage:\n";
    double x;
    vector<double> inps(2);
    unsigned i=0;
    while(cin>>x){
        inps[i++]=x;
        if(i==2){
            i=0;
            cout<<model.predict(inps)<<'\n';
        }
    }
    // cout<<model;

    return 0;
}