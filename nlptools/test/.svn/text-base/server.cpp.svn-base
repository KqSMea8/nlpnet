#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>

#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <map>
#include<algorithm>


#include "NNRewrite.h"
#include "ConcRewrite.h"
#include "MLPRewrite.h"
#include "ConvRewrite.h"
#include "EnsembleMLP.h"
#include "LRRewrite.h"
#include "utils.h"

#define MAX_TEXT_LENG 12048
#define DEFAULT_PORT 8851

/* index of different results in buffer. */
#define SYN_TERM_INDEX 0
#define SYN_QUERY_INDEX 1
#define TOPN_REWRITE_INDEX 2
#define EXPRESSION_INDEX 3

char result[MAX_TEXT_LENG];

#define MAX_FILE_PATH   1024
#define MAX_TERM_COUNT  1024
typedef pair<string, real> PAIR;

using namespace std;


bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {
    return lhs.second > rhs.second; 
}

void sort_map(map<string,real>& dict, vector<PAIR>& kv)
{
    kv.clear();
    kv.insert(kv.begin(), dict.begin(), dict.end());
    sort(kv.begin(), kv.end(),cmp_by_value);
}

string model_file;
NNRewrite* pModel;
vector<PAIR> kv;
map<string,real> sub_dict;
map<string,real> z2_dict;
int top_num=30;
void rewrite(vector<string> query, nn_pack& pack, ostringstream& oss)
{
    for(size_t i=0; i<query.size(); i++){
        if(query[i]=="L1" || query[i]=="L2" || query[i]=="R1" || query[i]=="R2")
        {
            continue;
        }
       sub_dict.clear();
       z2_dict.clear();
       pModel->Predict(query,i,sub_dict,pack,0);
       pModel->Predict(query,i,z2_dict,pack,1);
       kv.clear();
       kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
       sort(kv.begin(), kv.end(),cmp_by_value);
       for(size_t j=0; j<kv.size(); j++){
           if(j>top_num) break;
           oss << kv[j].first << "\t" << kv[j].second << "\t"<< z2_dict[kv[j].first] << endl;
       }
       if(sub_dict.size() >0){
           oss <<" #############################################" << endl;
       }
    }
}

int main(int argc, char* argv[]) 
{
    if (argc == 1) {
        printf("Options:\n");
        printf("Parameters for test:\n");
        printf("\t-model-file <file>\n");
        printf("\t\t load the model to predict \n");
        return -1;
    }
    int i;
    if ((i = ArgPos((char *)"-model-file", argc, argv)) > 0) model_file=argv[i + 1];
    pModel = new MLPRewrite();
    nn_pack pack;
    pModel->InitPack(pack);
    pModel->LoadModel(model_file);

    cerr << "load the model done" << endl;

    signal(SIGPIPE, SIG_IGN);
    // init the socket.
    char inbuf[MAX_TEXT_LENG] ={ 0 };
    int sockfd = -1, newfd = -1;
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "[error] socket creation failed; errno: %d",errno);
        exit(-1);
    }

    struct sockaddr_in clientAddr;
    struct sockaddr_in serverAddr;
    bzero(&serverAddr, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddr.sin_port = htons(DEFAULT_PORT);
    if (bind(sockfd, (struct sockaddr*) &serverAddr, sizeof(struct sockaddr))< 0)				//whj:绑定
    {
        fprintf(stderr, "[error] socket binding failed; errno: %d",errno);
        exit(-1);
    }

    listen(sockfd, 1000);
    fprintf(stderr, "start to listen ...\n");

    char str_lang[256]={0};
    char str_query[256]={0};

    string line;
    vector<string> query;
    ostringstream oss;
    while(1)
    {
        // waiting for receiving a connection.
        int size = sizeof(struct sockaddr_in);
        newfd = accept(sockfd, (struct sockaddr*) &clientAddr,(socklen_t*) &size);						//whj：accept
        fprintf(stderr,"accept over\n");

        if (newfd < 0)
        {
            fprintf(stderr,"error: accept() error, errno is %d",errno);
            close(newfd);
            continue;
        }
        fprintf(stderr, "received a request from %s\n",inet_ntoa(clientAddr.sin_addr));


        while(1)
        {
            memset(inbuf,0,sizeof(inbuf));
            memset(result,0,sizeof(result));
            // read text from client.
            int len = read(newfd, inbuf, MAX_TEXT_LENG);	//whj: read
            if (len <= 0)
            {
                fprintf(stderr, "error: read() error in server. errno: %d\n",errno);
                break;
            }
            inbuf[len] = 0;

            while (len > 0 && (inbuf[len - 1] == '\r' || inbuf[len - 1] == '\n'))
                len -= 1;
            inbuf[len] = 0;

            fprintf(stderr,"inbuf: %s\n",inbuf);
            line.assign(inbuf);
            cerr << "query:" << line << endl;
            split_str(query, line, " ");
            oss.str("");
            rewrite(query, pack, oss);
            sprintf(result,"################## Rewrite Result ################### \n%s",oss.str().c_str());
            // write the result to socket.
            len = write(newfd, result, sizeof(result));
            if (len < 0)
            {
                //fprintf(stderr, "error: write() error in server\n");
                fprintf(stderr, "[error] write() failed; errno: %d",errno);
                break;
            }
            fprintf(stderr, "writed %d bytes to client\n", len);
        }

        close(newfd);

    }

    close(sockfd);



    return 0;
}

