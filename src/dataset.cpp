#include "dataset.hpp"
#include "utils.hpp"

#include <ctime>
#include <cstdlib>

ReadInfo DataSet::m_read_info;
WriteInfo DataSet::m_write_info;

DataSet::DataSet(const std::string& filepath, const ReadInfo& read_info){
    load(filepath, read_info);
}

DataSet::DataSet(const Matrix_d& matrix, unsigned beg_row, unsigned end_row, const std::vector<std::string>& labels){
    m_labels=labels;
    load(matrix, beg_row, end_row);
}

DataSet::~DataSet(){}

Matrix_d DataSet::matrix() const{
    return m_matrix;
}

std::vector<Matrix_d> DataSet::compile(unsigned n_output) const{
    std::vector<Matrix_d> matrices(2);
    
    matrices[0]=m_matrix.compile(0, m_matrix.shape().n_col-n_output);
    matrices[1]=m_matrix.compile(m_matrix.shape().n_col-n_output, m_matrix.shape().n_col);

    return matrices;
}

void DataSet::set_labels(const std::vector<std::string>& labels){
    m_labels=labels;
}

std::vector<std::string> DataSet::labels() const{
    return m_labels;
}

std::string DataSet::labels_string() const{
    std::string labels;
    for(unsigned i=0;i<m_labels.size();++i){
        labels+=m_labels[i];
        if(i<m_labels.size()-1) labels+=',';
    }
    return labels;
}

std::vector<DataSet> DataSet::split(const std::initializer_list<double> percentages){
    std::vector<DataSet> datasets;
    unsigned beg, end, size=0;
    auto percentage=percentages.begin();

    for(unsigned i=0;percentage<percentages.end();++percentage){
        size=m_matrix.shape().n_row*(*percentage)/100.;
        beg=i;
        end=beg+size;
        if(end>m_matrix.shape().n_row) end=m_matrix.shape().n_row;
        if(percentage+1==percentages.end() && end!=m_matrix.shape().n_row)
            end=m_matrix.shape().n_row;
        i=end;
        datasets.push_back(DataSet(m_matrix, beg, end, m_labels));

        if(end==m_matrix.shape().n_row) break;
    }

    return datasets;
}

void DataSet::shuffle(){
    srand(time(0));

    unsigned prev, next;
    unsigned n_row=m_matrix.shape().n_row;
    for(unsigned i=0;i<n_row;++i){
        prev=i;
        next=abs((rand()/(double)RAND_MAX)*n_row);
        m_matrix.swap(prev, next);
    }
}

Shape DataSet::shape() const{
    return m_matrix.shape();
}

bool DataSet::empty() const{
    return (m_matrix.shape().n_row==0);
}

void DataSet::dc_sort(){
    recursive_sort(0, 0, m_matrix.shape().n_row);
}

double DataSet::biased(){
    double bias=-1*min(*this)+1;

    if(bias<1) return 0.;

    for(unsigned i=0;i<m_matrix.shape().n_row;++i){
        for(unsigned j=0;j<m_matrix.shape().n_col-1;++j)
            m_matrix[i][j]+=bias;
    }
    return bias;
}

void DataSet::load(const std::string& filepath, const ReadInfo& info){
    unsigned n_row=0, n_col=0;
    Shape shape;
    
    std::stringstream stream;
    std::string line, token;
    std::ifstream file(filepath);

    if(!file){
        std::cout<<"ERROR [loading file]: Couldn't open up the file!\n";
        exit(1);
    }

    if(!info.shape_skip){
        std::getline(file, line);
        std::stringstream stream(line);

        std::getline(stream, token, ',');
        shape.n_row=std::stoi(token);
        std::getline(stream, token, ',');
        shape.n_col=std::stoi(token);
    }
    if(!info.header_skip){
        std::getline(file, line);
        std::stringstream stream(line);
        while(std::getline(stream, token, ',')){
            m_labels.push_back(token);
        }
    }

    while(std::getline(file, line)){
        if(line.empty() && info.empty_line_skip) continue;

        stream<<line;
        m_matrix.add_row();
        while(std::getline(stream, token, ',')){
            m_matrix[n_row].push_back(std::stod(token));
            ++n_col;
        }

        if( (shape.n_col && shape.n_col!=n_col) || 
            (n_row && n_col!=m_matrix.shape().n_col) ){
            std::cout<<"ERROR [file loading]: Different dimensions!\n";
            exit(1);
        }

        m_matrix.set_shape(m_matrix.shape().n_row, n_col);
        n_col=0; ++n_row;
        stream.str(std::string());
        stream.clear();
    }

    if(shape.n_row && shape.n_row!=n_row){
        std::cout<<"WARNING [loading file]: Shape doesn't match!\n";
    }

    file.close();
}

void DataSet::load(const Matrix_d& matrix, unsigned beg_row, unsigned end_row){
    if(beg_row>matrix.shape().n_row || end_row>matrix.shape().n_row){
        std::cout<<"ERROR [matrix loading]: Row limit exceeded!\n";
        exit(1);
    }

    if(!end_row) end_row=matrix.shape().n_row;
    m_matrix.reshape(end_row-beg_row, matrix.shape().n_col);

    for(unsigned i=beg_row;i<end_row;++i){
        for(unsigned j=0;j<matrix.shape().n_col;++j){
            m_matrix[i-beg_row][j]=matrix.get(i,j);
        }
    }
}

void DataSet::save(const std::string& filepath, const WriteInfo& info) const{
    std::ofstream file(filepath);
    
    if(!file){
        std::cout<<"ERROR [saving dataset]: Couldn't open up the file!\n";
        exit(1);
    }

    if(!info.shape_skip){
        file<<m_matrix.shape()<<std::endl;
    }
    if(!info.header_skip && !m_labels.empty()){
        for(unsigned i=0;i<m_labels.size();++i){
            file<<m_labels[i];
            if(i<m_labels.size()-1) file<<',';
        }   file<<std::endl;
    }

    for(unsigned i=0;i<m_matrix.shape().n_row;++i){
        for(unsigned j=0;j<m_matrix.shape().n_col;++j){
            file<<m_matrix.get(i,j);
            if(j<m_matrix.shape().n_col-1) file<<", ";
        }   file<<std::endl;
    }

    file.close();
}

void DataSet::recursive_sort(unsigned col, unsigned beg_row, unsigned end_row){
    if(col>=m_matrix.shape().n_col || beg_row>=end_row) return ;

    sort_column(m_matrix, col, beg_row, end_row);
    for(;beg_row<end_row && col<m_matrix.shape().n_col-1;){
        unsigned tmp_end_row=diff_index(m_matrix, col, beg_row, end_row);
        recursive_sort(col+1, beg_row, tmp_end_row);
        beg_row=tmp_end_row;
    }
}

std::ostream& operator<<(std::ostream& out, const DataSet& dataset){
    out<<dataset.shape()<<'\n';
    for(unsigned i=0;i<dataset.m_matrix.shape().n_row;++i){
        for(unsigned j=0;j<dataset.m_matrix.shape().n_col;++j){
            out<<dataset.m_matrix.get(i,j)<<'\t';
        }   out<<'\n';
    }
    return out;
}