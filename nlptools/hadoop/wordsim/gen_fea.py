#coding:gbk
import sys
from itertools import combinations

trigram={}
slot_tag="XXX"
min_cnt=5
def map(line):
    items=line.split()
    size=len(items)
    for i in range(size-2):
        tri="|".join(items[i:i+3])
        if tri not in trigram:
            trigram[tri]=1
        else:
            trigram[tri]+=1


def map_out():
    for k,v in trigram.items():
        print k+"\t"+str(v)

def reduce_out():
    for k,v in trigram.items():
        if v>min_cnt:
            tri=k.split("|")
            f0="X|"+tri[1]+"|"+tri[2]
            f1=tri[0]+"|X|"+tri[2]
            f2=tri[0]+"|"+tri[1]+"|X"
            print  tri[0]+"\t"+f0+"\t"+str(v)
            print  tri[1]+"\t"+f1+"\t"+str(v)
            print  tri[2]+"\t"+f2+"\t"+str(v)

def reduce(line):
    items=line.split("\t")
    if items[0] not in trigram:
        trigram[items[0]]=int(items[1])
    else:
        trigram[items[0]]+=int(items[1])

    
if __name__ == '__main__':
    cnt=0
    while True:
        try:
            line=raw_input().strip()
            if cnt%100000==0:
                sys.stderr.write("linenum:"+str(cnt)+"\n")
            cnt+=1
            if sys.argv[1]=="map":
                map(line)
            else:
                reduce(line)
        except EOFError:
            break
    if sys.argv[1]=="map":
        map_out()
    else:
        reduce_out()
