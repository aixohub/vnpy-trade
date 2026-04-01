# vnpy-trade



## 命令一览


| 命令 | 功能 | 示例 | 说明                                                |
|---|---| ---|---------------------------------------------------|
| `/fbo` | 买入期权（同 `buyoption` 格式） | /fbo sndk put  600 4/2 1 lmt 1 | /futu_buyoption <标的> <put or call> <行权价> <到期日> <数量> <lmt 限价> | 
| `/fso` | 卖出持仓（期权/股票） |/fso sndk put  600 4/2 1 lmt 1 |/futu_sell <标的> <put or call> <行权价> <到期日> <数量> <lmt 限价> |
| `/flo [open\|all]` | 查询订单，含撤单按钮 | /flo all |
| `/fpos` | 持仓 + 未实现盈亏 | /fpos |
| `/facc` | 账户净值/现金/购买力 |/facc |
| `/ftradelog` | 交易日志（CSV + JSON 持久化） |/ftradelog |

---