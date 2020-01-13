
#ifndef _MATRIX_
#define _MATRIX_

#include <vector>
#include <iostream>
#include <functional>
#include <string>

#include "shape.hpp"

#define Matrix_d    Matrix<double>
#define Matrix_f    Matrix<float>
#define Matrix_l    Matrix<long>
#define Matrix_i    Matrix<int>
#define Matrix_u    Matrix<unsigned>
#define Matrix_c    Matrix<char>


// using fp = void (*)(Matrix&);

// Function Type
enum FT{
    READ=0,
    WRITE,
    COMPARE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    MULTIPLY_COEFFICIENT,
    MULTIPLY_ELEMENTWISE
};

template<class T=double>
class Matrix{
    private:
    Shape m_shape=Shape{0,0};
    std::vector<std::vector<T>> m_elems;

    public:
    Matrix()=default;
    Matrix(const Shape& shape);
    Matrix(unsigned n_row, unsigned n_col);
    Matrix(unsigned n_diog);
    ~Matrix();

    T get(unsigned row, unsigned col) const;
    std::vector<T> get(unsigned row) const;
    void initialize(T beg, T end);
    void set(const FT& ft, std::function<void(Matrix& mat, void* argument)>& function);
    void reshape(const Shape& shape);
    void reshape(unsigned n_row, unsigned n_col=0);
    void add_row(unsigned n_row=1);
    void add_col(unsigned n_col=1);
    void set_shape(unsigned n_row, unsigned n_col);
    void swap(unsigned row1, unsigned row2);
    Shape shape() const;
    Matrix<T> compile(unsigned beg_col, unsigned end_col) const;

    static Matrix<T> transpose(const Matrix<T>& mat);

    Matrix<T>& transpose();
    Matrix<T> mult(const Matrix& mat) const;
    Matrix<T> mul(const T& coef) const;
    Matrix<T> mulew(const Matrix<T>& mat) const;
    Matrix<T>& mult_eq(const Matrix<T>& mat);
    Matrix<T>& mul_eq(const T& coef);
    Matrix<T>& mulew_eq(const Matrix<T>& mat);

    std::vector<T>& operator[](unsigned index);
    Matrix<T> operator+(const Matrix<T>& mat) const;
    Matrix<T> operator-(const Matrix<T>& mat) const;
    Matrix<T> operator*(const Matrix<T>& mat) const;
    Matrix<T>& operator+=(const Matrix<T>& mat);
    Matrix<T>& operator-=(const Matrix<T>& mat);
    Matrix<T>& operator*=(const Matrix<T>& mat);

    template<class C>
    friend std::ostream& operator<<(std::ostream& out, const Matrix<C>& mat);
    template<class C>
    friend std::istream& operator>>(std::istream& in, Matrix<C>& mat);

    private:
    void (*m_functions[7])(Matrix&, void*);

    void read(const std::string& filepath);
    void write(const std::string& filepath) const;
    Matrix<T>& add(const Matrix& mat);
    Matrix<T>& subtract(const Matrix& mat);
    Matrix<T>& multiply(const Matrix& mat);
    Matrix<T>& multiply_coefficient(const T& coef);
    Matrix<T>& multiply_elementwise(const Matrix<T>& mat);

    void init();
};

template<class C>
std::ostream& operator<<(std::ostream& out, const Matrix<C>& mat);
template<class C>
std::istream& operator>>(std::istream& in, Matrix<C>& mat);

#ifndef _MATRIX_CPP_
#include "../src/matrix.cpp"
#endif

#endif
