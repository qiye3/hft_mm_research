"""标准行情的逐记录质量标签公共运算，供Data来源整理使用。"""

import polars as pl


# 【非核心】将各项布尔检查结果组成逐记录质量标签列表。
def build_record_flags(rules: dict[str, pl.Expr]) -> pl.Expr:
    """输入标签名与布尔表达式，返回 List(String) 类型的标签表达式。

    未命中的条件产生空列表，条件为空值时按未命中处理。
    调用方负责为缺失字段设置明确规则，本函数只组织标签，保留所有输入行。
    """
    return pl.concat_list([
        pl.when(condition.fill_null(False)).then(pl.lit(name)).otherwise(None)
        for name, condition in rules.items()
    ]).list.drop_nulls()
