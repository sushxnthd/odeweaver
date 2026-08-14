import argparse, json
import numpy as np
from .core import fit

def main():
    p=argparse.ArgumentParser(); p.add_argument('file'); p.add_argument('--dt',type=float,required=True); p.add_argument('--degree',type=int,default=2); p.add_argument('--threshold',type=float,default=0.05)
    a=p.parse_args(); x=np.load(a.file); models=fit(x,a.dt,degree=a.degree,threshold=a.threshold)
    print(json.dumps([m.equation(f'x{i}') for i,m in enumerate(models)],indent=2))

if __name__ == "__main__":
    main()
