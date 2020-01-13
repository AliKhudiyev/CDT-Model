
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <initializer_list>

#include "matrix.hpp"

#define INC         false
#define INCLUDE     false
#define EXC         true
#define EXCLUDE     true

#define IO_SPLIT    true
#define NO_SPLIT    false

#define READ(n) \
    (ReadInfo((n&512)>>3, (n&64)>>2, (n&8)>>1, n&1))

#define WRITE(n)    \
    (WriteInfo((n&8)>>1, n&1))

struct ReadInfo{
    bool shape_skip         = true;
    bool header_skip        = false;
    bool empty_line_skip    = true;
    bool empty_value_skip   = true;

    ReadInfo()=default;
    ReadInfo(int ss, int hs, int els, int evs): 
        shape_skip(ss), header_skip(hs), empty_line_skip(els), empty_value_skip(evs) {}
};

struct WriteInfo{
    bool shape_skip         = true;
    bool header_skip        = true;

    WriteInfo()=default;
    WriteInfo(int ss, int hs): 
        shape_skip(ss), header_skip(hs) {}
};

class DataSet{
    private:
    std::string m_filepath;
    std::vector<std::string> m_labels;
    Matrix_d m_matrix;

    public:
    DataSet()=default;
    DataSet(const std::string& filepath, const ReadInfo& read_info=DataSet::m_read_info);
    DataSet(const Matrix_d& matrix, unsigned beg_row=0, unsigned end_row=0, const std::vector<std::string>& labels=std::vector<std::string>());
    ~DataSet();

    Matrix_d matrix() const;
    std::vector<Matrix_d> compile(unsigned n_output=1) const;

    void set_labels(const std::vector<std::string>& labels);
    std::vector<std::string> labels() const;
    std::string labels_string() const;
    std::vector<DataSet> split(const std::initializer_list<double> percentages);
    void shuffle();
    Shape shape() const;
    bool empty() const;

    void load(const std::string& filepath, const ReadInfo& info=DataSet::m_read_info);
    void load(const Matrix_d& matrix, unsigned beg_row=0, unsigned end_row=0);
    void save(const std::string& filepath, const WriteInfo& info=DataSet::m_write_info) const;

    friend std::ostream& operator<<(std::ostream& out, const DataSet& dataset);

    private:
    static ReadInfo m_read_info;
    static WriteInfo m_write_info;
};

std::ostream& operator<<(std::ostream& out, const DataSet& dataset);
