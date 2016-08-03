#coding:gbk
import sys
import math

def map(line):
    items=line.split("\t")
    adj=items[1].split("|")
    for a in adj:
        print a+"\t"+items[0]+"\t"+items[2]

######################################
lines=[]
key=""
def reduce(line):
    global key
    global lines
    items=line.split("\t")
    if items[0]!=key:
        if len(lines)>1:
            flist1=lines[0][2].split("|")
            dict1={}
            for j in range(len(flist1)/2):
                dict1[flist1[j*2]]=float(flist1[2*j+1])

            for i in range(1,len(lines)):
                dict2={}
                flist2=lines[i][2].split("|")
                for j in range(len(flist2)/2):
                    dict2[flist2[2*j]]=float(flist2[2*j+1])
                cosine=cal_cosine(dict1,dict2)
                print key+"\t"+lines[i][1]+"\t"+str(cosine)
        lines=[0]
        key=items[0]

    if items[0]==items[1]:
        lines[0]=items
    else:
        lines.append(items)


def cal_cosine(dict1,dict2):
    a=0;
    b=0;
    ab=0;
    for k in dict1:
        a+=dict1[k]*dict1[k]
        if k in dict2:
            ab+=dict1[k]*dict2[k]
            
    for k in dict2:
        b+=dict2[k]*dict2[k]
    if a==0 or b==0:
        print dict1
        print dict2
    return ab/(math.sqrt(a)*math.sqrt(b))

    
if __name__ == '__main__':
    while True:
        try:
            line=raw_input().strip()
            if sys.argv[1]=="reduce":
                reduce(line)
            if sys.argv[1]=="map":
                map(line)
        except EOFError:
            break
    #push the last one
    if sys.argv[1]=="reduce":
        reduce("a\ta|b|c\tf1|1.0|f2|1.0")
