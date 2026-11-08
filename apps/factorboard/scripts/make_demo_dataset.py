"""Create visibly labeled synthetic *input data*, for smoke testing the full evaluation pipeline.
Not representative of real BTCUSDT trading or model performance.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def make(out:Path,days=7,points=1500,seed=47):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    start=pd.Timestamp('2026-09-01T00:00:00Z').value
    for day in range(days):
        n=points;ts=start+day*86_400_000_000_000 + np.arange(n,dtype=np.int64)*100_000_000
        raw=rng.normal(size=n)
        book=np.empty(n);book[0]=raw[0]
        for j in range(1,n):book[j]=.91*book[j-1]+.09*raw[j]
        imb=np.clip(book*.45,-.94,.94)
        trade=np.clip(.65*imb+rng.normal(0,.35,n),-1,1)
        futureshock=.008*(np.roll(imb,1))+rng.normal(0,.028,n)
        mid=95_000+day*135+np.cumsum(futureshock)
        frame=pd.DataFrame({'timestamp_ns':ts,'book_segment_id':np.zeros(n,dtype=int),
          'trade_segment_id':np.zeros(n,dtype=int),'book_feature_valid':np.ones(n,dtype=bool),
          'trade_feature_valid':np.ones(n,dtype=bool),'book_valid':np.ones(n,dtype=bool),
          'imbalance_1':np.clip(imb+rng.normal(0,.05,n),-1,1),
          'imbalance_5':imb, 'imbalance_10':np.clip(.85*imb+rng.normal(0,.04,n),-1,1),
          'trade_imbalance_100ms':trade,'mid_price':mid,'spread_ticks':np.ones(n)})
        buy=np.maximum(0,(1+trade)*(.2+np.abs(rng.normal(0,.4,n))))
        sell=np.maximum(0,(1-trade)*(.2+np.abs(rng.normal(0,.4,n))))
        bn=pd.Series(buy).rolling(10,min_periods=10).sum().to_numpy()
        sn=pd.Series(sell).rolling(10,min_periods=10).sum().to_numpy()
        frame['buy_volume_1s']=bn;frame['sell_volume_1s']=sn
        frame['trade_imbalance_1s']=(bn-sn)/(bn+sn)
        frame['return_1s_bps']=(mid/np.r_[np.full(10,np.nan),mid[:-10]]-1)*1e4
        for ms in [100,500,1000,5000]:
            step=ms//100
            fut=np.r_[mid[step:],np.full(step,np.nan)]
            frame[f'y_mid_{ms}ms_bps' if ms!=1000 else 'y_mid_1s_bps']=(fut/mid-1)*1e4
        # Label naming: 100ms, 500ms, 1s, 5s. No label crosses midnight.
        frame.to_csv(out/f'{pd.Timestamp(start+day*86_400_000_000_000,unit="ns",tz="UTC").strftime("%Y-%m-%d")}.csv',index=False,float_format='%.12g')
    (out/'synthetic_manifest.json').write_text(json.dumps({'synthetic':True,'purpose':'smoke test only','seed':seed,'days':days,'points_per_day':points},ensure_ascii=False,indent=2),encoding='utf8')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='workspace/data/demo');p.add_argument('--days',type=int,default=7);p.add_argument('--points',type=int,default=1500)
    a=p.parse_args();print(make(Path(a.out),a.days,a.points))
