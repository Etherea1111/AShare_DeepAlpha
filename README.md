# AShare-DeepAlpha

A 股日频量价多因子选股研究原型。当前工作聚焦于单一时点数据管线、可复现基线模型和验证期成本敏感回测；测试集与最终样本外集尚未用于模型选择。

## 当前范围

- 数据源：BaoStock，J66「货币金融服务」行业，45 只股票。
- 数据区间：2018-01-02 至 2024-12-31，原始日频行情未复权。
- 特征：21 个量价和截面特征，包括动量、波动率、成交量、换手率、Amihud、RSI 和市场状态。
- 复权：因子和标签使用由 `close/preclose` 构造的总收益指数，扣除分红送转造成的原始价格跳变。
- 标签：未来 5 日和 20 日总收益调整后的收益。
- 模型：方向因子组合、Ridge、Elastic Net、距离加权 KNN 和逻辑回归方向模型。
- 回测：验证期 Top-10 组合，t 日收盘生成信号，t+1 开盘成交，模拟涨跌停、停牌、整手、手续费、印花税和滑点。

项目名称中的深度学习模型尚未实现。当前状态是研究基线和回测验证阶段，不代表策略已经具备实盘价值。

## 环境

需要 Python 3.10+。安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

核心清洗、因子、模型和回测脚本主要使用标准库；NumPy 用于加速 KNN，Pandas 和 BaoStock 仅用于重新下载原始数据。也可以使用项目内 `.venv`。

## 数据切分

| 数据集 | 日期范围 | 交易日数 | 行数 |
| --- | --- | ---: | ---: |
| 训练集 | 2019-01-10 至 2023-03-17 | 1,015 | 37,416 |
| 验证集 | 2023-03-20 至 2023-10-24 | 145 | 6,519 |
| 测试集 | 2023-10-25 至 2024-05-30 | 145 | 6,525 |
| 最终样本外集 | 2024-05-31 至 2024-12-31 | 145 | 6,525 |

训练集、验证集、测试集和最终样本外集严格按交易日切分。测试集和最终样本外集在当前阶段保持未读取。

## 复现

在项目根目录执行：

```bash
python3 scripts/clean_industry_data.py
python3 scripts/split_cleaned_data.py
python3 scripts/build_train_factors.py
python3 scripts/fit_model_baselines.py
python3 scripts/analyze_factor_stability.py
python3 scripts/backtest_validation.py
```

`fit_model_baselines.py` 会执行训练期 purged walk-forward 选择、验证集预测和模型报告生成，当前环境约需 9 分钟。`backtest_validation.py` 约需 1 秒。

历史点时成分股下载需要网络和 BaoStock 接口：

```bash
python3 scripts/download_historical_constituents.py
python3 scripts/build_point_in_time_panel.py
```

在历史快照文件成功生成前，现有研究结果仍使用当前行业成分股快照，存在幸存者偏差。当前 BaoStock 连接会在查询后断开，下载脚本尚未完成全区间快照。

## 产物

| 产物 | 路径 |
| --- | --- |
| 原始行情 | `data/j66_money_finance_daily_2018_2024.csv` |
| 清洗行情 | `data/cleaned/j66_money_finance_daily_2018_2024.csv` |
| 状态台账 | `data/cleaned/j66_money_finance_status_2018_2024.csv` |
| 数据集切分 | `data/cleaned/splits/` |
| 训练因子与标签 | `data/processed/train_factor_labels.csv` |
| 验证预测 | `data/processed/validation_predictions_5d.csv`、`validation_predictions_20d.csv` |
| 模型报告 | `data/processed/model_baseline_report.json` |
| 因子 IC 汇总 | `data/analysis/factor_ic_summary.csv` |
| 因子分年度 IC | `data/analysis/factor_ic_yearly.csv` |
| 因子两两相关性 | `data/analysis/factor_correlation_pairs.csv` |
| 因子稳定性报告 | `data/analysis/factor_ic_report.json` |
| 验证期回测 | `data/backtest/validation_backtest_summary.json` |
| 回测净值和交易 | `data/backtest/validation_backtest_nav.csv`、`validation_backtest_trades.csv` |
| 阶段报告 | `reports/项目阶段报告.md` |
| 版本日志 | `reports/版本更新日志.md` |

## 因子 IC 稳定性

`analyze_factor_stability.py` 按交易日计算各因子与未来 5/20 日收益的截面 Spearman RankIC，汇总年度表现，并用 `horizon - 1` 阶 Newey-West 滞后处理重叠标签。它还会计算因子两两截面相关性，供后续降维和组合构建参考。

当前训练期与验证期结果：

| 期限 | 可用因子 | 同号因子 | 两期均值 IC Spearman | 相对稳定的方向 |
| --- | ---: | ---: | ---: | --- |
| 5 日 | 19 | 16 | 0.498 | `close_location`、短中期波动率、20 日换手率均为负；量比和成交额比为弱正 |
| 20 日 | 19 | 12 | 0.055 | 波动率、换手率和 `close_location` 仍偏负，但多数动量/反转类因子发生符号切换 |

`market_return_1` 和 `market_breadth_up` 在同一交易日内对所有股票相同，无法计算截面 RankIC，因此在汇总中标记为 `no_cross_sectional_variation`。5 日因子的横截面排序在两期之间更稳定；20 日因子的稳定性和符号一致性明显较弱。

## 最新验证结果

当前 `model_baseline_report.json` 使用 21 个特征、5,608 条验证记录。主要结果：

| 期限 | 模型 | 日均 RankIC | Top/Bottom 20% 收益差 |
| --- | --- | ---: | ---: |
| 5 日 | 方向因子组合 | 0.0439 | 23.01 bps/日 |
| 5 日 | KNN (`k=60`) | 0.0310 | 24.30 bps/日 |
| 20 日 | 方向因子组合 | 0.0702 | 33.22 bps/日 |
| 20 日 | KNN (`k=5`) | 0.0083 | 14.61 bps/日 |

验证期 Top-10 组合在基准成本假设（手续费 3 bps、印花税 5 bps、滑点 10 bps）下全部为负收益。5 日等权行业组合收益为 +0.68%，20 日为 -2.20%。复权后方向因子组合的预测能力明显提高，但仍不足以覆盖换手成本。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖因子未来数据隔离、标签边界、因子 RankIC 与 Newey-West 计算、分年度汇总、KNN 加速与标准库回退一致性、模型数值逻辑、点时成员退出事件和回测成交规则。

## 已知限制

- 仅覆盖单一行业和有限验证期，结论不能外推到全 A 股。
- 总收益指数由 `close/preclose` 构造，能处理分红送转，但尚未与供应商直接提供的完整复权 OHLC 逐笔核验。
- 历史点时成分股尚未成功下载，当前存在幸存者偏差。
- 涨跌停和上市状态为规则估算，尚未接入完整历史交易规则库。
- 回测是验证期诊断，不是实盘交易系统；未模拟订单排队、冲击成本和容量约束。
- 深度学习基线和最终样本外评估尚未完成。

## 下一步

1. 取得历史点时成分股并重跑数据、因子、模型和回测。
2. 与供应商复权 OHLC 核验总收益指数的公司行为处理。
3. 冻结验证方案后评估测试集，最终样本外集保持到最后一次性使用。
4. 增加树模型和深度学习模型，并与当前基线和等权组合比较。
5. 补齐行业归因、更多成本场景和组合层面的因子稳定性检验。
