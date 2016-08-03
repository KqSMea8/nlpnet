#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ostream>
#include <algorithm>
#include "utils.h"
#include "NNRewrite.h"
using namespace std;

void toBinary(const string& infile, const string& outfile) {

    ifstream fin1(infile.c_str());
    string line;
    vector<string> vec;
    vector<string> str_tmp;

    getline(fin1, line);
    split_str(vec, line, "\t");
    int vocab_size = atoi(vec[1].c_str());
    int word_vec_size = atoi(vec[2].c_str());

    getline(fin1, line);
    split_str(vec, line, "\t");
    int sub_num = atoi(vec[1].c_str());
    int sub_vec_size = atoi(vec[2].c_str());

    getline(fin1, line);
    split_str(vec, line, "\t");
    int w_num = atoi(vec[1].c_str());

    getline(fin1, line);
    split_str(vec, line, "\t");
    int b_num = atoi(vec[1].c_str());

    FILE *fo;
    fo = fopen(outfile.c_str(), "wb");
    fprintf(fo, "%d %d %d %d %d %d\n", vocab_size, word_vec_size, sub_num,
        sub_vec_size, w_num, b_num);

    int sub_end = sub_num + vocab_size;
    int cnt = 0;
    while (getline(fin1, line)) {
        split_str(vec, line, "\t");
        if (cnt < vocab_size) {
            fprintf(fo, "%s ", vec[0].c_str());
            for (int b = 1; b < vec.size(); b++) {
                real tmp = atof(vec[b].c_str());
                fwrite(&tmp, sizeof(real), 1, fo);
            }
            fprintf(fo, "\n");
        }
        if (cnt >= vocab_size && cnt < sub_end) {
            //string key = vec[1] + "|" + vec[2];
            //cerr << key << endl;
            fprintf(fo, "%s ", vec[1].c_str());
            fprintf(fo, "%s ", vec[2].c_str());
            real tmp = atof(vec[0].c_str());
            fwrite(&tmp, sizeof(real), 1, fo);
            for (int b = 3; b < vec.size(); b++) {
                real tmp = atof(vec[b].c_str());
                fwrite(&tmp, sizeof(real), 1, fo);
            }
            fprintf(fo, "\n");
        }
        if (cnt >= sub_end) {
            string key="weight";
            fprintf(fo, "%s ", key.c_str());
            for (int b = 0; b < vec.size(); b++) {
                real tmp = atof(vec[b].c_str());
                fwrite(&tmp, sizeof(real), 1, fo);
            }
            fprintf(fo, "\n");
        }
        cnt+=1;
    }
}

void toTxt(const string& infile) {

    FILE *f;
    int vocab_size, word_vec_size, sub_num, sub_vec_size, w_num, b_num;
    char ch;
    float *M;
    char *vocab;
    char *vocab2;

    f = fopen(infile.c_str(), "rb");
    fscanf(f, "%d", &vocab_size);
    fscanf(f, "%d", &word_vec_size);
    fscanf(f, "%d", &sub_num);
    fscanf(f, "%d", &sub_vec_size);
    fscanf(f, "%d", &w_num);
    fscanf(f, "%d", &b_num);

    cout<< "#wordvec\t" << vocab_size <<"\t" <<word_vec_size << endl;
    cout<< "#subvec\t" << sub_num <<"\t" <<sub_vec_size << endl;
    cout<< "#w_num\t" << w_num<< endl;
    cout<< "#b_num\t" << b_num<< endl;


    int sub_end=sub_num+vocab_size;
    int w_end=sub_num+vocab_size+w_num;
    int b_end=sub_num+vocab_size+w_num+b_num;
    int w_size=6*word_vec_size*sub_vec_size;
    vocab = (char *)malloc( 1024 * sizeof(char));
    vocab2 = (char *)malloc( 1024 * sizeof(char));
    M = (float *)malloc(1000000 * sizeof(float));
    for (int b = 0; b < b_end; b++) {
        if (b < vocab_size) {
            fscanf(f, "%s%c", vocab, &ch);
            fread(&M[0], sizeof(float), 1, f);
            cout<<string(vocab) << "\t" << int(M[0]);
            for (int a = 1; a < word_vec_size + 1; a++) {
                fread(&M[a], sizeof(float), 1, f);
                cout <<  "\t" << M[a];
            }
            cout << endl;
        }

        if (b >= vocab_size && b < sub_end) {
            fscanf(f, "%s%c", vocab, &ch);
            //cerr << string(vocab) << endl;
            fscanf(f, "%s%c", vocab2, &ch);
            //cerr << string(vocab) << endl;
            fread(&M[0], sizeof(float), 1, f);
            cout << int(M[0]) << "\t" << string(vocab) << "\t" << string(vocab2);
            for (int a = 2; a < sub_vec_size + 2; a++) {
                fread(&M[a], sizeof(float), 1, f);
                cout << "\t" << M[a];
            }
            cout << endl;
        }

        if (b >= sub_end && b < w_end) {
            fscanf(f, "%s%c", vocab, &ch);
            fread(&M[0], sizeof(float), 1, f);
            cout << int(M[0]);
            for (int a = 1; a < w_size + 1; a++) {
                fread(&M[a], sizeof(float), 1, f);
                cout << "\t" << M[a];
            }
            cout << endl;

        }

        if (b >= w_end) {
            fscanf(f, "%s%c", vocab, &ch);
            fread(&M[0], sizeof(float), 1, f);
            cout << int(M[0]);
            for (int a = 1; a < sub_vec_size + 1; a++) {
                fread(&M[a], sizeof(float), 1, f);
                cout << "\t" << M[a];
            }
            cout << endl;
        }

    }
    fclose(f);
}



int main(int argc, char **argv) {


    if(argc<2){
        cerr << "Usage: ./" << argv[0] << " 0 infile outfile" << endl;
        cerr << "Usage: ./" << argv[0] << " 1 infile " << endl;
        return 0;
    }
    int mode=atoi(argv[1]);
    if(mode==0){
        toBinary(argv[2], argv[3]);
    }

    if(mode==1){
        toTxt(argv[2]);
    }
    return 0;
}

