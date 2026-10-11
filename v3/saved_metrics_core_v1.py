"""Independent fullC unordered-diploid point metrics; exact zeros/missing separate."""
import numpy as np

def neural(logits,truth):
 x=np.asarray(logits,dtype=np.float64);t=np.asarray(truth,dtype=np.int64)
 if x.ndim!=2 or x.shape[0]!=len(t)*2 or x.shape[1]<2 or not np.isfinite(x).all() or t.shape!=(len(t),2) or ((t< -1)|(t>=x.shape[1])).any():raise ValueError('whole finite ordered fullC logits and diploid namespace required')
 z=x-x.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));p=np.exp(lp).reshape(-1,2,x.shape[1]);lp=lp.reshape(p.shape);valid=(t>=0).all(1);a=np.maximum(t[:,0],0);b=np.maximum(t[:,1],0);ix=np.arange(len(t));l1=lp[ix,0,a]+lp[ix,1,b];l2=lp[ix,0,b]+lp[ix,1,a];logq=np.where(a==b,l1,np.logaddexp(l1,l2));prob=np.exp(logq);norm=np.sum(p[:,0]**2,1)*np.sum(p[:,1]**2,1)+np.sum(p[:,0]*p[:,1],1)**2-np.sum((p[:,0]*p[:,1])**2,1);dose=p.sum(1);true=np.zeros_like(dose);np.add.at(true,(ix,a),1);np.add.at(true,(ix,b),1);m=np.stack((-logq,norm-2*prob+1,np.mean((dose[:,1:]-true[:,1:])**2,axis=1)),axis=1);m[~valid]=np.nan;state=np.where(valid,0,1).astype('u1');dose[~valid]=np.nan
 if not np.isfinite(m[valid]).all():raise ValueError('stable fullC neural metrics must remain finite')
 return state,m,dose

def baselines(counts,copies,truth,NLL,Brier,states):
 c=np.asarray(counts);t=np.asarray(truth);copy=np.asarray(copies);n=len(t);C=len(c)
 if c.shape!=(C,) or C<2 or (c<0).any() or copy.shape!=t.shape or t.shape!=(n,2) or ((t< -1)|(t>=C)).any() or ((copy< -1)|(copy>=C)).any():raise ValueError('whole exact baseline namespace')
 ds=np.full((2,n,C),np.nan);metrics=np.full((2,n,3),np.nan);valid=(t>=0).all(1);true=np.zeros((n,C));ix=np.arange(n);np.add.at(true,(ix,np.maximum(t[:,0],0)),1);np.add.at(true,(ix,np.maximum(t[:,1],0)),1)
 for j in range(2):
  st=states[j];expected=np.where(valid,0,1).astype('u1')
  if j==0:
   if c.sum()==0:expected[valid]=2
   else:
    f=c.astype(np.float64)/c.sum();ds[j,valid]=2*f;v=f[np.maximum(t[:,0],0)]*f[np.maximum(t[:,1],0)]*np.where(t[:,0]==t[:,1],1.,2.);expected[valid&(v==0)]=3
    finite=valid&(v>0);calc=np.full(n,np.nan);calc[finite]=-np.log(v[finite]);calc[expected==3]=np.inf;norm=2*np.dot(f,f)**2-np.sum(f**4);br=np.where(valid,norm-2*v+1,np.nan)
  else:
   av=valid&(copy>=0).all(1);expected[valid&~av]=2;match=(np.sort(copy,axis=1)==np.sort(t,axis=1)).all(1);expected[av&~match]=3;calc=np.where(av,np.where(match,0.,np.inf),np.nan);br=np.where(av,np.where(match,0.,2.),np.nan)
   for row in np.flatnonzero(av):ds[j,row]=0;np.add.at(ds[j,row],copy[row],1)
  if not np.array_equal(st,expected):raise ValueError('baseline full missing/zero states differ')
  av=np.isin(st,[0,3])
  if av.any():
   if not np.allclose(NLL[j,av],calc[av],atol=1e-12,rtol=1e-12) or not np.allclose(Brier[j,av],br[av],atol=1e-12,rtol=1e-12):raise ValueError('independent original baseline point agreement differs')
   metrics[j,:,0]=NLL[j];metrics[j,:,1]=Brier[j];metrics[j,av,2]=np.mean((ds[j,av,1:]-true[av,1:])**2,1)
  if not np.isnan(NLL[j,~av]).all() or not np.isnan(Brier[j,~av]).all():raise ValueError('unavailable baseline became numeric zero')
 return np.asarray(states,dtype='u1'),metrics,ds

def allele_r2(dosages,truth,common):
 d=np.asarray(dosages,dtype=np.float64);t=np.asarray(truth);C=d.shape[2];ix=np.flatnonzero(common);out=np.full((len(d),C-1),np.nan)
 if len(ix)<2:return out
 y=np.zeros((len(ix),C));np.add.at(y,(np.arange(len(ix)),t[ix,0]),1);np.add.at(y,(np.arange(len(ix)),t[ix,1]),1);y=y[:,1:];y-=y.mean(0);sy=np.sum(y*y,0)
 for k in range(len(d)):
  x=d[k,ix,1:].copy()
  if not np.isfinite(x).all():raise ValueError('whole common available dosage vector required')
  x-=x.mean(0);sx=np.sum(x*x,0);cov=np.sum(x*y,0);ok=(sy>0)&(sx>0);out[k,ok]=(cov[ok]/np.sqrt(sx[ok]*sy[ok]))**2
 return out
