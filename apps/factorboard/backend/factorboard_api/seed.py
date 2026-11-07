"""Only known Gen001, Gen002, and 16 active feature definitions; never generate fake factor metrics."""
from __future__ import annotations

GENS = [
    {'id':'Gen001','title':'买卖压力持续水平','family':'rolling_mean','hypothesis':'压力的持续水平是否预测未来收益？','status':'已设计'},
    {'id':'Gen002','title':'买卖压力短长窗口差','family':'diff_delta','hypothesis':'压力的近期变化是否具有预测增量？','status':'已设计'},
]

BASES = [
 ('book_obi_l1', 'book', '一档买卖盘压力', ('imbalance_1',), 'identity', ('bid_quantity_1', 'ask_quantity_1')),
 ('book_obi_l5', 'book', '五档买卖盘压力', ('imbalance_5',), 'identity', tuple(f'{side}_quantity_{i}' for side in ('bid','ask') for i in range(1,6))),
 ('book_obi_l10','book','十档买卖盘压力', ('imbalance_10',), 'identity', tuple(f'{side}_quantity_{i}' for side in ('bid','ask') for i in range(1,11))),
 ('trade_aggr_imb_100ms','trade','100ms主动成交失衡',('trade_imbalance_100ms',),'identity',('buy_volume','sell_volume')),
 ('trade_aggr_imb_1s','trade','1s主动成交失衡',('trade_imbalance_1s',),'identity',('buy_volume_1s','sell_volume_1s')),
 ('cross_trade_book_gap','cross','成交与盘口压力背离',('trade_imbalance_1s','imbalance_5'),'difference',('buy_volume_1s','sell_volume_1s','bid_quantity_1..5','ask_quantity_1..5')),
]
FEATURES = [
 ('spread_ticks','book','当前最优买卖价差（Tick）','spread','Tick'),
 ('imbalance_1','book','一档买卖盘失衡','L1 bid/ask depth','ratio'),
 ('imbalance_5','book','五档买卖盘失衡','L5 bid/ask depth','ratio'),
 ('imbalance_10','book','十档买卖盘失衡','L10 bid/ask depth','ratio'),
 ('bid_depth_10','book','十档买方总挂单量','bid_quantity_1..10','BTC'),
 ('ask_depth_10','book','十档卖方总挂单量','ask_quantity_1..10','BTC'),
 ('microprice_minus_mid','book','微价格相对中间价偏移','BBO price and size','USDT/BTC'),
 ('buy_volume_1s','trade','过去1s主动买入量','trade side and size','BTC'),
 ('sell_volume_1s','trade','过去1s主动卖出量','trade side and size','BTC'),
 ('buy_volume_5s','trade','过去5s主动买入量','trade side and size','BTC'),
 ('sell_volume_5s','trade','过去5s主动卖出量','trade side and size','BTC'),
 ('trade_imbalance_100ms','trade','过去100ms主动成交失衡','buy_volume / sell_volume','ratio'),
 ('trade_imbalance_1s','trade','过去1s主动成交失衡','buy_volume_1s / sell_volume_1s','ratio'),
 ('return_100ms_bps','returns','过去100ms中间价收益','mid_price','bps'),
 ('return_1s_bps','returns','过去1s中间价收益','mid_price','bps'),
 ('return_5s_bps','returns','过去5s中间价收益','mid_price','bps'),
]

def seed_factor_definitions():
    factors=[]
    for bname, category,desc, cols,inner,raw in BASES:
        for gen,op,short,long in [('Gen001','mean',10,None),('Gen002','mean_diff',10,50)]:
            suffix = f'mean_{short}' if long is None else f'mean_diff_{short}_{long}'
            fid=f'{bname}_{suffix}'
            factors.append(dict(id=fid,gen=gen,category=category,desc=desc,base_name=bname,base_columns=list(cols),base_op=inner,operation=op,short_window=short,long_window=long,post='raw',raw_fields=list(raw),status='候选',unit='ratio'))
    for name,category,desc,formula,unit in FEATURES:
        factors.append(dict(id=name,gen='FEATURES',category=category,desc=desc,base_name=name,base_columns=[name],base_op='identity',operation='identity',short_window=None,long_window=None,post='raw',raw_fields=[formula],status='候选',unit=unit))
    return factors
