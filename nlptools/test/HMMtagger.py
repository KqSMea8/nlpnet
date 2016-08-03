import sys
from collections import defaultdict
import math
import datetime

vocab=defaultdict(lambda:-1)
emit_prob=[]
trans_prob={}
tag_num=1000
min_value=-100000000000.0

def load_emit_prob(vocab_file,cluster_file):
    index=0
    for line in open(vocab_file):
        vocab[line.strip()]=index
        index+=1

    for line in open(cluster_file):
        items=line.strip().split() 
        vec=[]
        for i in range(len(items)):
            if i%2==0:
                vec.append(int(items[i]))
            else:
                vec.append(math.log(float(items[i])))
        emit_prob.append(vec) 

    sys.stderr.write("load_emit_prob done\n")

def cal_trans_prob(train_file):
    trigram={}
    #Pseudo-count
    alpha=0.001
    for line in open(train_file):
        items=line.strip().split()
        for i in range(len(items)-2):
            tri="\t".join(items[i:i+3])
            flag=0
            for w in items[i:i+3]:
                if w not in vocab:
                    flag=1
            if flag==1: continue
            if tri not in trigram:
                trigram[tri]=1
            else:
                trigram[tri]+=1

    for tri in trigram:
        ss=tri.split("\t")
        for i in range(len(ss)):
            ss[i]=emit_prob[vocab[ss[i]]][0]
        bi=(ss[0],ss[1])
        if bi not in trans_prob:
            trans_prob[bi]={}
        if ss[2] not in trans_prob[bi]:
            trans_prob[bi][ss[2]]=1
        else:
            trans_prob[bi][ss[2]]+=1


    for bi in trans_prob:
        sum=tag_num*alpha
        for p in trans_prob[bi]:
            sum+=trans_prob[bi][p]
        for p in trans_prob[bi]:
            trans_prob[bi][p]=math.log((trans_prob[bi][p]+alpha)/sum)
        trans_prob[bi][-1]=math.log(alpha/sum) 

    sys.stderr.write("cal_trans_prob done\n")

def save_trans_prob(outfile):        
    out=open(outfile,'w')
    for bi in trans_prob:
        for p in trans_prob[bi]:
            line=str(bi[0])+"\t"+str(bi[1])+"\t"+str(p)+"\t"+str(trans_prob[bi][p])
            out.write(line+"\n")
            #print line

def decode(train_file,out_file):
    id=0
    out=open(out_file,'w')
    starttime = datetime.datetime.now()
    for line in open(train_file):
        if id%10000==0:
            sys.stderr.write("progress:"+str(id)+"\n")
            endtime = datetime.datetime.now()
            print (endtime-starttime).seconds
        id+=1
        items=line.split()
        tags=decode_one(items)
        ss=[]
        for i in range(len(items)): 
            ss.append(items[i]+"_"+str(tags[i]))
        out.write("\t".join(ss)+"\n")
    out.flush()
    out.close()


def get_states(x,k):
    list=[]
    if k<0:
        return [-1]
    id=vocab[x[k]]
    if id==-1:
        return [-1]
    for i in range(len(emit_prob[id])):
        if i%2==0:
            list.append(emit_prob[id][i])
    return list

def get_trans_prob(s1,s2,s3):
    if s1==-1 or s2==-1:
        return 0
    if (s1,s2) not in trans_prob:
        return min_value
    if s3 not in trans_prob[(s1,s2)]:
        return trans_prob[(s1,s2)][-1]
    return trans_prob[(s1,s2)][s3]
    
def decode_one(x):
    pi=defaultdict(lambda:0)
    bp={}
    n=len(x)
    for k in range(n):
        cur_states=get_states(x,k)
        pre_states=get_states(x,k-1)
        pre_pre_states=get_states(x,k-2)
        for i in range(len(cur_states)):
            s=cur_states[i]
            for u in pre_states:               
                pi[(k,u,s)]=2*min_value
                for t in pre_pre_states:
                    id=vocab[x[k]]
                    score=0
                    if id==-1:
                        score=pi[(k-1,t,u)]+get_trans_prob(t,u,s)
                    else:
                        score=pi[(k-1,t,u)]+get_trans_prob(t,u,s)+emit_prob[id][2*i+1]
                    print "####"+str(k)+":"+str(t)+","+str(u)+","+str(s)+":"+str(score)
                    if score>=pi[(k,u,s)]:
                        pi[(k,u,s)]=score
                        bp[(k,u,s)]=t

    t=[0]*len(x)
    max=2*min_value
    #print "##################### emit_prob"
    #print emit_prob
    #print "##################### trans_prob"
    #print trans_prob
    cur_states=get_states(x,n-1)
    pre_states=get_states(x,n-2)
    for s in cur_states:
        for u in pre_states:
            score=pi[(n-1,u,s)]
            #print str(u)+"\t"+str(s)+"\t"+str(score)
            if score>=max:
                max=score
                t[n-1]=s
                t[n-2]=u
    #print "##################### bp"
    #print bp
    #print "##################### tags"
    #print t
    for i in range(n-2):
        k=n-3-i
        t[k]=bp[(k+2,t[k+1],t[k+2])]
    assert len(x)==len(t)
    return t   

def decode_one(x):
    pi=defaultdict(lambda:0)
    bp={}
    n=len(x)
    for k in range(n):
        cur_states=get_states(x,k)
        pre_states=get_states(x,k-1)
        pre_pre_states=get_states(x,k-2)
        for i in range(len(cur_states)):
            s=cur_states[i]
            for u in pre_states:               
                pi[(k,u,s)]=2*min_value
                for t in pre_pre_states:
                    id=vocab[x[k]]
                    score=0
                    if id==-1:
                        score=pi[(k-1,t,u)]+get_trans_prob(t,u,s)
                    else:
                        score=pi[(k-1,t,u)]+get_trans_prob(t,u,s)+emit_prob[id][2*i+1]
                    #print "####"+str(k)+":"+str(t)+","+str(u)+","+str(s)+":"+str(score)
                    if score>=pi[(k,u,s)]:
                        pi[(k,u,s)]=score
                        bp[(k,u,s)]=t

    t=[0]*len(x)
    max=2*min_value
    cur_states=get_states(x,n-1)
    pre_states=get_states(x,n-2)
    for s in cur_states:
        for u in pre_states:
            score=pi[(n-1,u,s)]
            #print str(u)+"\t"+str(s)+"\t"+str(score)
            if score>=max:
                max=score
                t[n-1]=s
                t[n-2]=u
    for i in range(n-2):
        k=n-3-i
        t[k]=bp[(k+2,t[k+1],t[k+2])]
    assert len(x)==len(t)
    return t   


if __name__ == '__main__':

    if len(sys.argv)<4:
        sys.stderr.write("usage:"+sys.argv[0]+" vocab_file cluster_file train_file test-file test-out-file\n")
        exit(1)

    load_emit_prob(sys.argv[1],sys.argv[2])
    cal_trans_prob(sys.argv[3])
    save_trans_prob(sys.argv[3]+".trans")
    decode(sys.argv[4],sys.argv[5])
