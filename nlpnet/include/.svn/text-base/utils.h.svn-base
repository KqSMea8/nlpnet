/*
 * utils.h
 *
 *  Created on: 2013-6-26
 *      Author: zengzengfeng
 */

#ifndef UTILS_H_
#define UTILS_H_
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Eigen>
using namespace std;


namespace nlpnet {


// Status Code
enum StatusCode {
    SUCCESS = 0,
    FAILED = -1,
    ERROR = -2,
};


typedef double real_t;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> IMatrix;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, 1, Eigen::ColMajor> IVector;
typedef Eigen::DenseBase<Eigen::Matrix<real_t, -1, -1> >::ColXpr ColXpr;

//typedef std::shared_ptr<IMatrix> MatrixPtr;
//typedef std::shared_ptr<IVector> VectorPtr;

typedef IMatrix* MatrixPtr;
typedef IVector* VectorPtr;
void split_str(vector<string>& vecToken, const string& str, string separator);
int arg_pos(char *str, int argc, char **argv);
/*! \brief random initiailze array within a range */
inline void rand_init_array(real_t* array, size_t size, real_t min, real_t max) {
    for (size_t i = 0; i < size; ++i) {
        real_t a = rand() / (real_t) RAND_MAX;
        array[i] = a*(max - min) + min;
    }
}

const string MODEL_CONFIG = "model_config";

}

#endif /* UTILS_H_ */
