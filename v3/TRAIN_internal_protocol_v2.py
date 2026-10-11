"""TRAIN-only selection rules. Geometry, masks and all-candidate losses are fixed."""
import math

def partition_train(train,folds,heldout=3):
 validation=list(folds[heldout]);v=set(validation);fit=[r for r in train if r not in v]
 if not fit or not validation or set(fit)&v or set(fit)|v!=set(train):raise ValueError('whole disjoint TRAIN family fold required')
 return fit,validation,{k:list(r) for k,r in folds.items() if k!=heldout}

def scheduled_lr(base,step,total,warmup=.05,minimum=.1):
 if total<1 or step<0 or step>=total:raise ValueError('exact schedule step range required')
 warm=max(1,math.ceil(total*warmup))
 if step<warm:return base*(step+1)/warm
 progress=(step-warm)/max(1,total-warm-1)
 return base*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress)))

def choose_configuration(trials):
 complete=[t for t in trials if t.get('status')=='COMPLETE' and t.get('validation_family_count',0)>0 and math.isfinite(t['internal_selection_NLL'])]
 if len(complete)!=len(trials) or not complete:raise ValueError('all intended LR trials need complete finite internal-family validation')
 return min(complete,key=lambda t:(t['internal_selection_NLL'],t['learning_rate'],t['selected_epochs']['MASKED_PRETRAIN'],t['selected_epochs']['SNP_ONLY_IMPUTATION']))
