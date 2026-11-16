import numpy as np
from common import *

def kmeans(Z,k):
    n=len(Z); step=n//k; cent=Z[np.arange(k)*step].copy(); labels=np.full(n,-1)
    while True:
        dist=np.sum((Z[:,None,:]-cent[None,:,:])**2,axis=2); new=np.argmin(dist,axis=1)
        changed=np.any(new!=labels); labels=new
        for j in range(k):
            if np.any(labels==j): cent[j]=Z[labels==j].mean(axis=0)
        if not changed: break
    wcss=sum(np.sum((Z[labels==j]-cent[j])**2) for j in range(k))
    return labels,wcss

def hierarchy():
    names=['E1','E2','E3','E4','E5','E6']; D=np.array([[0,7.9,7.4,2.9,2.9,6.1],[7.9,0,1.4,5.9,5.3,2.0],[7.4,1.4,0,5.8,4.6,1.3],[2.9,5.9,5.8,0,2.8,4.5],[2.9,5.3,4.6,2.8,0,3.4],[6.1,2.0,1.3,4.5,3.4,0]],float)
    allrows=[]; cuts={}
    for method in ['single','complete','average']:
        clusters=[[i] for i in range(6)]; heights=[]
        for step in range(1,6):
            candidates=[]
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)):
                    vals=[D[x,y] for x in clusters[i] for y in clusters[j]]
                    h=min(vals) if method=='single' else max(vals) if method=='complete' else float(np.mean(vals))
                    candidates.append((h,i,j))
            h,i,j=min(candidates,key=lambda q:q[0]); A=clusters[i]; B=clusters[j]
            if min(A)>min(B): A,B=B,A
            sa='+'.join(names[x] for x in sorted(A)); sb='+'.join(names[x] for x in sorted(B))
            allrows.append([method,step,sa,sb,h]); heights.append(h)
            merged=sorted(A+B); clusters=[c for ix,c in enumerate(clusters) if ix not in (i,j)]+[merged]; clusters.sort(key=lambda c:min(c))
        gaps=np.diff(heights); largest=int(np.argmax(gaps)); cuts[method]=6-(largest+1)
    return allrows,cuts

def main():
    a,_=load(); X=np.column_stack([a[k] for k in ['account_age_months','failed_txn_count_90d','credit_utilisation_ratio','avg_monthly_inflow_inr_thousands','repayment_score']]); mu=X.mean(0); sd=X.std(0,ddof=1); Z=(X-mu)/sd; d={}; wc={}
    for k in range(2,7):
        labels,v=kmeans(Z,k); wc[k]=v; d[f'wcss_k{k}']=v
    elbow=next((k for k in (3,4,5) if wc[k]-wc[k+1] < .5*(wc[k-1]-wc[k])),5); d['elbow_k']=elbow
    labels,_=kmeans(Z,3); order=sorted(range(3),key=lambda j:X[labels==j,2].mean()); seglabels=np.zeros(800,dtype=int); segrows=[]
    for ix,j in enumerate(order):
        mask=labels==j; seglabels[mask]=ix; name='ABC'[ix]; vals=[name,int(mask.sum()),*X[mask].mean(0),float(a['loan_default'][mask].mean()),float(a['monthly_spend_inr_thousands'][mask].mean())]
        segrows.append(vals); d[f'size_segment_{name}']=int(mask.sum())
    write_csv('part3_segments.csv',['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'],segrows)
    hr,cuts=hierarchy(); write_csv('part3_hierarchical.csv',['linkage','merge_no','cluster_a','cluster_b','height'],hr)
    for k,v in cuts.items(): d[f'{k}_clusters_at_cut']=v
    S=(Z.T@Z)/(len(Z)-1); ev,V=np.linalg.eigh(S); ix=np.argsort(ev)[::-1]; ev=ev[ix]; V=V[:,ix]
    for j in range(5):
        if V[np.argmax(np.abs(V[:,j])),j]<0: V[:,j]*=-1
    ratio=ev/ev.sum(); cum=np.cumsum(ratio); d['eigenvalue_sum']=ev.sum()
    for i in range(5): d[f'eigenvalue_{i+1}']=ev[i]; d[f'explained_ratio_{i+1}']=ratio[i]; d[f'cumulative_ratio_{i+1}']=cum[i]
    d['components_for_80pct']=int(np.argmax(cum>=.8)+1)
    for i in range(5): d[f'pc1_loading_X{i+1}']=V[i,0]; d[f'pc2_loading_X{i+1}']=V[i,1]
    proj=Z@V[:,:2]
    for r in range(3): d[f'row{r+1}_pc1']=proj[r,0]; d[f'row{r+1}_pc2']=proj[r,1]
    for i,name in enumerate('ABC'): d[f'mean_pc1_segment_{name}']=proj[seglabels==i,0].mean()
    write_results('part3_results.txt',d)
if __name__=='__main__': main()
