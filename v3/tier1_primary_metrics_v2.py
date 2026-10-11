"""Long's preregistered primary dosage/concordance and secondary log scores.

Scalar probability flooring changes only the reported secondary log score.
It never renormalizes a distribution or changes calls, dosage or original NLL.
Undefined variance and unavailable methods remain explicit null endpoints.
"""
import math
NLL_FLOOR=1e-6

def secondary_log_scores(log_probability,floor=NLL_FLOOR):
 if floor!=NLL_FLOOR:raise ValueError('one preregistered identical floor required')
 if log_probability is None:return dict(unsmoothed_NLL=None,floored_log_score=None,state='UNAVAILABLE')
 if math.isnan(log_probability) or log_probability>1e-12:raise ValueError('valid exact event log probability required')
 return dict(unsmoothed_NLL=-log_probability,floored_log_score=-max(log_probability,math.log(floor)),state='EXACT_ZERO' if log_probability==-math.inf else 'AVAILABLE')

def TRAIN_frequency_bin(allele_count,reference_population_alleles):
 if reference_population_alleles is None or reference_population_alleles==0:return 'UNKNOWN_TRAIN_FREQUENCY'
 if not 0<=allele_count<=reference_population_alleles:raise ValueError('exact complete-phased TRAIN count required')
 f=allele_count/reference_population_alleles;minor=min(f,1-f)
 if minor==0:return '0'
 for upper,label in [(.001,'(0,0.1%]'),(.005,'(0.1%,0.5%]'),(.01,'(0.5%,1%]'),(.05,'(1%,5%]'),(.5,'(5%,50%]')]:
  if minor<=upper:return label
 raise AssertionError('minor frequency outside universe')

def dosage_r2(predicted,truth):
 if len(predicted)!=len(truth) or any(not math.isfinite(v) for v in [*predicted,*truth]):raise ValueError('same finite whole-allele paired dosage cohort required')
 n=len(truth)
 if n<3:return dict(r2=None,n=n,state='INSUFFICIENT_DONORS')
 px=math.fsum(predicted)/n;ty=math.fsum(truth)/n;xx=math.fsum((x-px)**2 for x in predicted);yy=math.fsum((y-ty)**2 for y in truth);xy=math.fsum((x-px)*(y-ty) for x,y in zip(predicted,truth))
 if xx==0 or yy==0:return dict(r2=None,n=n,state='CONSTANT_PREDICTION_OR_TRUTH')
 correlation=(xy/math.sqrt(xx))/math.sqrt(yy);value=correlation*correlation
 if value < -1e-12 or value>1+1e-12:raise ValueError('unstable covariance result')
 return dict(r2=value,n=n,state='DEFINED_PAIRED_CONDITIONAL')

def diploid_concordance(call,truth):
 if call is None or truth is None:return None
 if len(call)!=2 or len(truth)!=2 or any(x<0 for x in [*call,*truth]):raise ValueError('same complete unordered diploid namespace required')
 return int(sorted(call)==sorted(truth))
