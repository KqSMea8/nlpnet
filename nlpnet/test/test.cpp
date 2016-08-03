// string constructor
#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include  "optimizer.h"
#include "utils.h"
using namespace Eigen;
using namespace std;
typedef Matrix<double, Dynamic, Dynamic> MatrixXr;
int main (int argc, char** argv)
{

    /*std::string strValue = "{\"key1\":\"value1\"}";
    Json::Reader reader;
    Json::Value value;

    if (reader.parse(strValue, value))
    {
        std::string out = value["key1"].asString();
        std::cout << out << std::endl;
    }
    MatrixXr R=MatrixXr::Random(3,1);
    cout << "R=" << endl << R << endl;
    return 0;*/


    /*Json::Reader reader;// ½âjsonÓJson::Reader   
    Json::Value root; // Json::ValueÊһÖºÜØªµÄà£¬¿ÉԴúÒÀÐ¡£Èint, string, object, array         

    std::ifstream is;  
    is.open (argv[1], std::ios::binary );    
    if (reader.parse(is, root, false)){
        for(unsigned int i=0; i<root.size(); i++){
            cout << root[i]["layer_name"].asString() << endl;
        }
    } */

    Eigen::VectorXf aVector( 5 );
    aVector << 3, 4, 5, 6, 7;
    cout << aVector.maxCoeff();

    /*std::string strValue = "{\"key1\":\"value1\"}";
    Json::Reader reader;
    Json::Value value;

    if (reader.parse(strValue, value))
    {
        std::string out = value["key1"].asString();
        std::cout << out << std::endl;
    }
    MatrixXr R=MatrixXr::Random(3,1);
    cout << "R=" << endl << R << endl;*/

}
