# 180秒新模型确认长周期入场

以家族均衡180秒XGBoost的预测方向确认E4或compact6的1800秒费用信号。主模型多头时要求短周期预测大于0，空头时要求小于0；零或缺失短周期预测不准入。仅影响新入场，持仓期间继续使用原policy5风险退出，不因短周期反向直接平仓。

该180秒模型使用旧146列加123列家族均衡多尺度特征，已完成fit/tune训练，开发IC0.03723。本轮不重训、不搜索确认阈值；输入预测只读取timestamp_ns和prediction_log_bps，严格匹配主回放整秒时间键，不读取未来标签或evaluation_valid筛选交易。

两个开发案例均为0.125BTC、本金10000USDT、最长持仓3600秒、双边Taker 5bps及实际资金费。原无确认结果为E4净收益343.33USDT、compact6净收益551.05USDT，直接复用为对照。先比较实际净收益、交易次数、回撤、敞口及周/月集中度，暂不进入历史复核。

实现account_sizing.py的directional_entry_filter（核心416）；测试覆盖同向、反向、零、NaN和时间键错位。配置strategy/short_confirmation_20261114.yaml、research/short_confirmation_20261114.yaml，输出workspace/results/short_confirmation_20261114_v1。

本轮也查阅了既有60/120天近期拟合实验：八例均为开发亏损，最好年化约-9.02%，因此没有重复相同训练窗口实验。

## 完整开发结果

E4确认后18笔，净收益175.47USDT、年化12.73%、最高敞口0.9703、去最好周57.96USDT，未晋级。compact6确认后25笔，净收益435.40USDT、年化34.11%、最高敞口0.9766、回撤1.85%、去最好周234.68USDT；较原规则收益下降、回撤小幅下降。只冻结compact6进入相同历史复核，具体规则见short_confirmation_freeze.md，不重选参数。
