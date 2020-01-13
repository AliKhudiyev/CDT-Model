
#pragma once

#include <iostream>

struct Shape{
    unsigned n_row=0, n_col=0;
    bool is_diogonal;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;
    friend std::ostream& operator<<(std::ostream& out, const Shape& shape);
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
