#coding:gbk
import sys
import math

######################################
key=""
targets=set()
fea_dicts={}
def reduce(line):
    global key
    items=line.split("\t")
    if key!=items[0]:
        targets.add(key)
        adj="|".join(list(targets))
        fea=""
        for k,v in fea_dicts.items():
            fea+=k+"|"+str(v)+"|"
        if key!="":
            print key+"\t"+adj+"\t"+fea
        targets.clear()
        fea_dicts.clear()
        key=items[0]

    if len(items)==2:
        targets.add(items[1])
    else:
        fea_dicts[items[1]]=float(items[2])


    
if __name__ == '__main__':
    while True:
        try:
            line=raw_input().strip()
            reduce(line)
        except EOFError:
            break
    #push the last one
    reduce("word1\tfeature\t1.0")
