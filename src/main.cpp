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

    DataSet dataset(EXAMPLE("IRIS.csv"), READ(01011));
    dataset.shuffle();

    std::vector<DataSet> datasets=dataset.split({70,30});
    cout<<datasets[1]<<'\n';

    CDT model(datasets[0], 4, true);
    model.optimize(Optimization{NO_FIT, THIRD, 5});
    model.compile();
    // model.train();
    model.fit();

    // cout<<"\n > Testing stage:\n";
    // double x;
    // unsigned dim=dataset.shape().n_col-1;
    // vector<double> inps(dim);
    // unsigned i=0;
    // while(cin>>x){
    //     inps[i++]=x;
    //     if(i==dim){
    //         i=0;
    //         cout<<model.predict(inps)<<'\n';
    //     }
    // }
    model.test(cout, datasets[0]);
    model.test(cout, datasets[1]);
    // cout<<model;

    return 0;
}