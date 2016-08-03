/*
 * file_spliter.h
 *
 *  Created on: 2016-3-31
 *      Author: zengzengfeng
 */

#ifndef FILE_SPLITER_H_
#define FILE_SPLITER_H_
#include <iostream>
#include <fstream>
#include <string>
namespace nlpnet {

class FileSpliter {
public:
    FileSpliter(const std::string& file_name, int split_num, int index);
    ~FileSpliter();
    int getline(std::string& line);
private:
    std::ifstream *_fin_ptr;
    long long _size;
    long long _cur_size;
};

} /* namespace nlpnet */
#endif /* FILE_SPLITER_H_ */
