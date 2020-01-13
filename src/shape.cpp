#include "shape.hpp"

bool Shape::operator==(const Shape& shape) const{
    return (n_row==shape.n_row && n_col==shape.n_col);
}

bool Shape::operator!=(const Shape& shape) const{
    return (n_row!=shape.n_row || n_col!=shape.n_col);
}

std::ostream& operator<<(std::ostream& out, const Shape& shape){
    out<<shape.n_row<<" x "<<shape.n_col;
    return out;
}