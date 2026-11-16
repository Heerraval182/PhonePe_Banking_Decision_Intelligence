import numpy as np
from common import *

def entropy(y):
    if len(y)==0:return 0.
    p=np.mean(y); return -sum(q*np.log2(q) for q in (p,1-p) if q>0)
def gini(y):
    if len(y)==0:return 0.
    p=np.mean(y); return 1-p*p-(1-p)*(1-p)
def main():
    a,_=load(); X,Z,y,tr,te,mu,sd=split_standard(a); yt=y[tr]; d={'root_entropy':entropy(yt),'root_gini':gini(yt)}
    x1=a['account_age_months']; x3=a['credit_utilisation_ratio']; x6=a['irregular_salary_flag']; x7=a['autopay_flag']
    conds=[x6==1,x7==1,x3>.5,x1<=24]; splitrows=[]; gains=[]
    for name,cond in zip(['S1','S2','S3','S4'],conds):
        yes=cond[tr]; yy=yt[yes]; no=yt[~yes]; n=len(yt); ig=entropy(yt)-(len(yy)*entropy(yy)+len(no)*entropy(no))/n; gg=gini(yt)-(len(yy)*gini(yy)+len(no)*gini(no))/n
        splitrows.append([name,len(yy),int(yy.sum()),len(no),int(no.sum()),ig,gg]); gains.append((name,ig,gg))
    write_csv('part4_splits.csv',['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'],splitrows)
    best=max(gains,key=lambda z:z[1])[0]; bestg=max(gains,key=lambda z:z[2])[0]; d.update(best_split_by_info_gain=best,best_split_by_gini_gain=bestg)
    j=['S1','S2','S3','S4'].index(best); cond=conds[j]; side=cond[tr]; rate_yes=float(yt[side].mean()); rate_no=float(yt[~side].mean()); d.update(stump_rate_yes=rate_yes,stump_rate_no=rate_no)
    w,b,_=logistic_fit(Z,y,tr,.1,1000); lp=sigmoid(Z[te]@w+b); logistic=(lp>=.3).astype(int)
    # Reproduce CV-selected K from Part 2.
    cv=[]
    for k in range(1,22,2):
        scores=[]
        for fold in range(5):
            val=np.arange(len(tr))%5==fold; ref=~val; pred=knn_predict(Z[tr][ref],y[tr][ref],Z[tr][val],k); scores.append(metrics(y[tr][val],pred)['f1'])
        cv.append((float(np.mean(scores)),k))
    bestk=min(cv,key=lambda q:(-q[0],q[1]))[1]; knp,knprob=knn_predict(Z[tr],y[tr],Z[te],bestk,True)
    stump_p=np.where(cond[te],rate_yes,rate_no); stump=(stump_p>=.3).astype(int); hard=((logistic+knp+stump)>=2).astype(int); soft=((lp+knprob+stump_p)/3>=.3).astype(int)
    votes=[]
    for name,pred in [('logistic',logistic),('knn',knp),('stump',stump),('hard_vote',hard),('soft_vote',soft)]:
        m=metrics(y[te],pred); votes.append([name,m['tp'],m['fp'],m['tn'],m['fn'],m['precision'],m['recall'],m['f1']])
    write_csv('part4_voting.csv',['model','tp','fp','tn','fn','precision','recall','f1'],votes)
    # AdaBoost, two weak rules.
    yy=np.array([1,1,1,-1,-1,-1,1,-1]); h1=np.array([1,1,-1,-1,-1,-1,-1,-1]); h2=np.array([-1,1,1,-1,1,-1,1,-1]); weights=np.ones(8)/8; boostrows=[]; alphas=[]; errs=[]
    for rnd,h in [(1,h1),(2,h2)]:
        before=weights.copy(); err=float(before[h!=yy].sum()); alpha=.5*np.log((1-err)/err); un=before*np.exp(-alpha*yy*h); weights=un/un.sum(); errs.append(err); alphas.append(alpha)
        for i in range(8): boostrows.append([rnd,i+1,before[i],un[i],weights[i]])
    ensemble=np.sign(alphas[0]*h1+alphas[1]*h2); d.update(round1_error=errs[0],round1_alpha=alphas[0],round2_error=errs[1],round2_alpha=alphas[1],ensemble_accuracy=float(np.mean(ensemble==yy)))
    write_csv('part4_adaboost.csv',['round','point','weight_before','weight_unnormalised','weight_after'],boostrows)
    d['oob_fraction']=(1-1/560)**560
    mlp=[784*16+16,16*16+16,16*10+10]; cnn=[(5*5*1+1)*6,(5*5*6+1)*16,400*32+32,32*10+10]
    # Seven required CSV rows, CNN layers include conv1, conv2, dense1, dense2.
    arch=[['mlp_hidden1',mlp[0]],['mlp_hidden2',mlp[1]],['mlp_output',mlp[2]],['cnn_conv1',cnn[0]],['cnn_conv2',cnn[1]],['cnn_dense1',cnn[2]],['cnn_dense2',cnn[3]]]
    write_csv('part4_architecture.csv',['layer','parameters'],arch); d.update(mlp_total=sum(mlp),cnn_total=sum(cnn),fc_layer_params=128*128*3*1024+1024,conv_layer_params=(3*3*3+1)*64,rnn_total=128*64+128*128+16*128+128+16)
    write_results('part4_results.txt',d)
if __name__=='__main__': main()
