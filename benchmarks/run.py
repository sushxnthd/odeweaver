import json
from pathlib import Path
import numpy as np
from odeweaver import fit
rng=np.random.default_rng(4); dt=0.002; n=3000; x=np.empty(n); x[0]=0.18
for i in range(n-1): x[i+1]=x[i]+dt*1.5*x[i]*(1-x[i]/2.0)
y=x + rng.normal(scale=0.0005,size=n)
m=fit(y[:,None],dt,degree=2,threshold=0.08)[0]; c=dict(zip(m.terms,m.coefficients))
out={"linear_coef":float(c['x0']),"quadratic_coef":float(c['x0*x0']),"linear_abs_error":abs(float(c['x0'])-1.5),"quadratic_abs_error":abs(float(c['x0*x0'])+0.75)}
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
