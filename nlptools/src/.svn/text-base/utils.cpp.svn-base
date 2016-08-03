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

void map_plus(map<string, float>& in_map, string& key, float value) {
	map<string, float>::iterator iter = in_map.find(key);
	if (iter != in_map.end()) {
		iter->second += value;
	} else {
		in_map[key] = value;
	}
}
int ArgPos(char *str, int argc, char **argv)
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
void randomize_list(vector<int>& pid_list){
	srand((unsigned) time(NULL));
	for(int i=0;i<pid_list.size();i++){
		int j = rand() % pid_list.size();
		if(i!=j){
		   	int tmp=pid_list[i];
		   	pid_list[i]=pid_list[j];
		   	pid_list[j]=tmp;
		}
	}
}
