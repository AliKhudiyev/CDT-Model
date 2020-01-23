
#pragma once

struct Metrics{
    unsigned n_tp, n_tn;
    unsigned n_fp, n_fn;

    double accuracy() const;
    double precision() const;
    double sensitivity() const;
    double specificity() const;
    double overfit_rate() const;
};
