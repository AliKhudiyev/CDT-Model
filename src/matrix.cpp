#ifndef _MATRIX_CPP_
#define _MATRIX_CPP_
#include "matrix.hpp"
#endif

#include <iostream>
#include <fstream>
#include <sstream>

template<typename T>
T get_random(T beg, T end){
    return (end-beg)*(rand()/(double)RAND_MAX)+beg;
}

// -----------------

template<class T>
Matrix<T>::Matrix(const Shape& shape){
    m_shape=shape;

    m_elems.resize(shape.n_row);
    for(auto& vec: m_elems)
        vec.resize(shape.n_col);
    init();
}

template<class T>
Matrix<T>::Matrix(unsigned n_row, unsigned n_col){
    m_shape={n_row, n_col};

    m_elems.resize(n_row);
    for(auto& vec: m_elems)
        vec.resize(n_col);
    init();
}

template<class T>
Matrix<T>::Matrix(unsigned n_diog){
    m_shape={n_diog, n_diog};

    m_elems.resize(n_diog);
    for(size_t i=0;i<n_diog;++i){
        m_elems[i].resize(n_diog);
    }
    init();
}

template<class T>
Matrix<T>::~Matrix<T>(){}

template<class T>
T Matrix<T>::get(unsigned row, unsigned col) const{
    if(row>=m_shape.n_row || col>=m_shape.n_col){
        std::cout<<"ERROR [get]: Row or column limit exceeded!\n";
        exit(1);
    }
    return m_elems[row][col];
}

template<class T>
std::vector<T> Matrix<T>::get(unsigned row) const{
    if(row>=m_shape.n_row){
        std::cout<<"ERROR [get]: Row limit exceeded!\n";
        exit(1);
    }
    return m_elems[row];
}

template<class T>
void Matrix<T>::initialize(T beg, T end){
    srand(time(0));

    for(unsigned i=0;i<m_shape.n_row;++i){
        for(unsigned j=0;j<m_shape.n_col;++j)
            m_elems[i][j]=get_random<T>(beg, end);
    }
}

template<class T>
void Matrix<T>::set(const FT& ft, std::function<void(Matrix& mat, void* argument)>& function){
    m_functions[(unsigned)ft]=function;
}

template<class T>
void Matrix<T>::reshape(const Shape& shape){
    if(!m_elems.empty()) m_elems.clear();
    m_shape=shape;

    m_elems.resize(shape.n_row);
    if(shape.n_col){
        for(auto& vec: m_elems)
            vec.resize(shape.n_col);
    }
}

template<class T>
void Matrix<T>::reshape(unsigned n_row, unsigned n_col){
    reshape(Shape{n_row, n_col});
}

template<class T>
void Matrix<T>::add_row(unsigned n_row){
    m_shape.n_row+=n_row;
    for(unsigned i=0;i<n_row;++i)
        m_elems.push_back(std::vector<T>());
}

template<class T>
void Matrix<T>::add_col(unsigned n_col){
    m_shape.n_col+=n_col;
    T elem;
    for(unsigned i=0;i<m_shape.n_row;++i)
        m_elems.push_back(elem);
}

template<class T>
void Matrix<T>::set_shape(unsigned n_row, unsigned n_col){
    m_shape.n_row=n_row;
    m_shape.n_col=n_col;
}

template<class T>
void Matrix<T>::swap(unsigned row1, unsigned row2){
    std::vector<T> mat_row=m_elems[row1];
    m_elems[row1]=m_elems[row2];
    m_elems[row2]=mat_row;
}

template<class T>
Shape Matrix<T>::shape() const{
    return m_shape;
}

template<class T>
Matrix<T> Matrix<T>::compile(unsigned beg_col, unsigned end_col) const{
    Matrix<T> matrix(m_shape.n_row, end_col-beg_col);

    for(unsigned i=0;i<matrix.m_shape.n_row;++i){
        for(unsigned j=beg_col;j<end_col;++j){
            matrix[i][j-beg_col]=m_elems[i][j];
        }
    }

    return matrix;
}

template<class T>
Matrix<T> Matrix<T>::transpose(const Matrix<T>& mat){
    Matrix<T> result=mat;
    result.transpose();
    return result;
}

template<class T>
Matrix<T>& Matrix<T>::transpose(){
    Matrix<T> result(m_shape.n_col, m_shape.n_row);
    for(size_t i=0;i<m_shape.n_row;++i){
        for(size_t j=0;j<m_shape.n_col;++j){
            result.m_elems[j][i]=m_elems[i][j];
        }
    }
    *this=result;
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::mult(const Matrix& mat) const{
    Matrix<T> result;
    // TO DO

    return result;
}

template<class T>
Matrix<T> Matrix<T>::mul(const T& coef) const{
    Matrix<T> result=*this;
    result.multiply_coefficient(coef);
    return result;
}

template<class T>
Matrix<T> Matrix<T>::mulew(const Matrix<T>& mat) const{
    Matrix<T> result=*this;
    result.multiply_elementwise(mat);
    return result;
}

template<class T>
Matrix<T>& Matrix<T>::mult_eq(const Matrix<T>& mat){
    // TO DO
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::mul_eq(const T& coef){
    return multiply_coefficient(coef);
}

template<class T>
Matrix<T>& Matrix<T>::mulew_eq(const Matrix<T>& mat){
    return multiply_elementwise(mat);
}

template<class T>
std::vector<T>& Matrix<T>::operator[](unsigned index){
    if(index>=m_shape.n_row){
        std::cout<<"ERROR [[] operator]: Index out of bound!\n";
        exit(1);
    }
    return m_elems[index];
}

template<class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& mat) const{
    Matrix<T> result=*this;
    result.add(mat);
    return result;
}

template<class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& mat) const{
    Matrix<T> result=*this;
    result.subtract(mat);
    return result;
}

template<class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& mat) const{
    Matrix<T> result=*this;
    result.mult(mat);
    return result;
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& mat){
    add(mat);
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& mat){
    subtract(mat);
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& mat){
    multiply(mat);
    return *this;
}

template<class C>
std::ostream& operator<<(std::ostream& out, const Matrix<C>& mat){
    std::cout<<mat.m_shape<<'\n';
    for(const auto& vec: mat.m_elems){
        for(const auto& val: vec){
            std::cout<<val<<"\t";
        }   std::cout<<std::endl;
    }
    return out;
}

template<class C>
std::istream& operator>>(std::istream& in, Matrix<C>& mat){
    // TO DO
    return in;
}

template<class T>
void Matrix<T>::read(const std::string& filepath){
    std::ifstream file(filepath);
    ;
    file.close();
}

template<class T>
void Matrix<T>::write(const std::string& filepath) const{
    std::ofstream file(filepath);
    ;
    file.close();
}

template<class T>
Matrix<T>& Matrix<T>::add(const Matrix& mat){
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [+ operator]: Shapes don't match!\n";
        exit(1);
    }
    // m_functions[2](*this, (void*)&mat);
    for(unsigned i=0;i<m_shape.n_row;++i){
        for(unsigned j=0;j<m_shape.n_col;++j)
            m_elems[i][j]+=mat.m_elems[i][j];
    }
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::subtract(const Matrix& mat){
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [- operator]: Shapes don't match!\n";
        exit(1);
    }
    // m_functions[3](*this, (void*)&mat);
    for(unsigned i=0;i<m_shape.n_row;++i){
        for(unsigned j=0;j<m_shape.n_col;++j)
            m_elems[i][j]-=mat.m_elems[i][j];
    }
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::multiply(const Matrix& mat){
    if(m_shape.n_col!=mat.m_shape.n_row){
        std::cout<<"ERROR [* operator]: Shapes don't match!\n";
        exit(1);
    }
    // m_functions[4](*this, (void*)&mat);
    Matrix<T> result(m_shape.n_row, mat.shape().n_col);
    for(size_t i=0;i<result.shape().n_row;++i){
        for(size_t j=0;j<result.shape().n_col;++j){
            for(size_t t=0;t<result.shape().n_col;++t)
                result[i][j]+=m_elems[i][t]*mat.m_elems[t][j];
        }
    }
    *this=result;
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::multiply_coefficient(const T& coef){
    // m_functions[5](*this, (void*)&coef);
    for(unsigned i=0;i<m_shape.n_row;++i){
        for(unsigned j=0;j<m_shape.n_col;++j)
            m_elems[i][j]*=coef;
    }
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::multiply_elementwise(const Matrix<T>& mat){
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [mulew]: Shapes don't match!\n";
        exit(1);
    }
    // m_functions[6](*this, (void*)&mat);
    for(unsigned i=0;i<m_shape.n_row;++i){
        for(unsigned j=0;j<m_shape.n_col;++j)
            m_elems[i][j]*=mat.m_elems[i][j];
    }
    return *this;
}

template<class T>
void Matrix<T>::init(){
    for(unsigned i=0;i<7;++i) m_functions[i]=nullptr;
}