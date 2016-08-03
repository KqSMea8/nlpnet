/*
 * utils.h
 *
 *  Created on: 2013-6-26
 *      Author: zengzengfeng
 */

#include "utils.h"
#include "string.h"
#include <sstream>
#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>

namespace nlpnet {

int GLOBAL_thread_num = 1;
void split_str(vector<string>& vecToken,const string& str, string separator)
{
    size_t pos1, pos2;
    string token;
    size_t len = separator.length();
    pos1 = pos2 = 0;
    vecToken.clear();
    while((pos2 = str.find(separator.c_str(), pos1)) != string::npos)
    {
        token = str.substr(pos1, pos2-pos1);
        pos1 = pos2+len;
        if(!token.empty())
        {
            vecToken.push_back(token);
        }
    }
    token = str.substr(pos1);
    if(!token.empty())
    {
        vecToken.push_back(token);
    }
}


int arg_pos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a]))
    {
        if (a == argc - 1)
        {
             printf("Argument missing for %s\n", str);
             exit(1);
        }
        return a;
    }
    return -1;
}

}
