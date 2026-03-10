import smtplib
import os
import traceback
from abc import ABC, abstractmethod
from email.message import EmailMessage
from queue import Empty, Queue
from threading import Thread
from typing import TypeVar
from collections.abc import Callable

from vnpy.event import Event, EventEngine
from .app import BaseApp
from .event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_QUOTE
)
from .gateway import BaseGateway
from .object import (
    CancelRequest,
    LogData,
    OrderRequest,
    QuoteData,
    QuoteRequest,
    SubscribeRequest,
    HistoryRequest,
    OrderData,
    BarData,
    TickData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    Exchange
)
from .constant import Status
from .setting import SETTINGS
from .utility import TRADER_DIR
from .converter import OffsetConverter
from .logger import logger, DEBUG, INFO, WARNING, ERROR, CRITICAL
from .locale import _


EngineType = TypeVar("EngineType", bound="BaseEngine")


class BaseEngine(ABC):
    """
    Abstract class for implementing a function engine.
    """

    @abstractmethod
    def __init__(
        self,
        main_engine: "MainEngine",
        event_engine: EventEngine,
        engine_name: str,
    ) -> None:
        """"""
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.engine_name: str = engine_name

    def close(self) -> None:
        """"""
        return


class MainEngine:
    """
    Acts as the core of the trading platform.
    """

    def __init__(self, event_engine: EventEngine | None = None) -> None:
        """"""
        if event_engine:
            self.event_engine: EventEngine = event_engine
        else:
            self.event_engine = EventEngine()
        self.event_engine.start()

        self.gateways: dict[str, BaseGateway] = {}
        self.engines: dict[str, BaseEngine] = {}
        self.apps: dict[str, BaseApp] = {}
        self.exchanges: list[Exchange] = []

        os.chdir(TRADER_DIR)    # Change working directory
        self.init_engines()     # Initialize function engines

    def add_engine(self, engine_class: type[EngineType]) -> EngineType:
        """
        Add function engine.
        """
        engine: EngineType = engine_class(self, self.event_engine)      # type: ignore
        self.engines[engine.engine_name] = engine
        return engine

    def add_gateway(self, gateway_class: type[BaseGateway], gateway_name: str = "") -> BaseGateway:
        """
        Add gateway.
        """
        # Use default name if gateway_name not passed
        if not gateway_name:
            gateway_name = gateway_class.default_name

        gateway: BaseGateway = gateway_class(self.event_engine, gateway_name)
        self.gateways[gateway_name] = gateway

        # Add gateway supported exchanges into engine
        for exchange in gateway.exchanges:
            if exchange not in self.exchanges:
                self.exchanges.append(exchange)

        return gateway

    def add_app(self, app_class: type[BaseApp]) -> BaseEngine:
        """
        Add app.
        """
        app: BaseApp = app_class()
        self.apps[app.app_name] = app

        engine: BaseEngine = self.add_engine(app.engine_class)
        return engine

    def init_engines(self) -> None:
        """
        Init all engines.
        """
        self.add_engine(LogEngine)

        oms_engine: OmsEngine = self.add_engine(OmsEngine)
        self.get_tick: Callable[[str], TickData | None] = oms_engine.get_tick
        self.get_order: Callable[[str], OrderData | None] = oms_engine.get_order
        self.get_trade: Callable[[str], TradeData | None] = oms_engine.get_trade
        self.get_position: Callable[[str], PositionData | None] = oms_engine.get_position
        self.get_account: Callable[[str], AccountData | None] = oms_engine.get_account
        self.get_contract: Callable[[str], ContractData | None] = oms_engine.get_contract
        self.get_quote: Callable[[str], QuoteData | None] = oms_engine.get_quote
        self.get_all_ticks: Callable[[], list[TickData]] = oms_engine.get_all_ticks
        self.get_all_orders: Callable[[], list[OrderData]] = oms_engine.get_all_orders
        self.get_all_trades: Callable[[], list[TradeData]] = oms_engine.get_all_trades
        self.get_all_positions: Callable[[], list[PositionData]] = oms_engine.get_all_positions
        self.get_all_accounts: Callable[[], list[AccountData]] = oms_engine.get_all_accounts
        self.get_all_contracts: Callable[[], list[ContractData]] = oms_engine.get_all_contracts
        self.get_all_quotes: Callable[[], list[QuoteData]] = oms_engine.get_all_quotes
        self.get_all_active_orders: Callable[[], list[OrderData]] = oms_engine.get_all_active_orders
        self.get_all_active_quotes: Callable[[], list[QuoteData]] = oms_engine.get_all_active_quotes
        self.update_order_request: Callable[[OrderRequest, str, str], None] = oms_engine.update_order_request
        self.convert_order_request: Callable[[OrderRequest, str, bool, bool], list[OrderRequest]] = oms_engine.convert_order_request
        self.get_converter: Callable[[str], OffsetConverter | None] = oms_engine.get_converter

        email_engine: EmailEngine = self.add_engine(EmailEngine)
        self.send_email: Callable[[str, str, str | None], None] = email_engine.send_email

    def write_log(self, msg: str, source: str = "MainEngine") -> None:
        """
        Put log event with specific message.
        """
        log: LogData = LogData(msg=msg, gateway_name=source)
        event: Event = Event(EVENT_LOG, log)
        self.event_engine.put(event)

    def get_gateway(self, gateway_name: str) -> BaseGateway | None:
        """
        Return gateway object by name.
        """
        gateway: BaseGateway | None = self.gateways.get(gateway_name, None)
        if not gateway:
            self.write_log(_("找不到底层接口：{}").format(gateway_name))
        return gateway

    def get_engine(self, engine_name: str) -> BaseEngine | None:
        """
        Return engine object by name.
        """
        engine: BaseEngine | None = self.engines.get(engine_name, None)
        if not engine:
            self.write_log(_("找不到引擎：{}").format(engine_name))
        return engine

    def get_default_setting(self, gateway_name: str) -> dict[str, str | bool | int | float] | None:
        """
        Get default setting dict of a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            return gateway.get_default_setting()
        return None

    def get_all_gateway_names(self) -> list[str]:
        """
        Get all names of gateway added in main engine.
        """
        return list(self.gateways.keys())

    def get_all_apps(self) -> list[BaseApp]:
        """
        Get all app objects.
        """
        return list(self.apps.values())

    def get_all_exchanges(self) -> list[Exchange]:
        """
        Get all exchanges.
        """
        return self.exchanges

    def connect(self, setting: dict, gateway_name: str) -> None:
        """
        Start connection of a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("连接登录 -> {}").format(gateway_name))

            gateway.connect(setting)

    def disconnect(self, gateway_name: str) -> None:
        self.gateway__ = """
        Start connection of a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("断开连接 -> {}").format(gateway_name))

    def subscribe(self, req: SubscribeRequest, gateway_name: str) -> None:
        """
        Subscribe tick data update of a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("订阅行情 -> {}：{}").format(gateway_name, req))

            gateway.subscribe(req)

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        """
        Send new order request to a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("委托下单 -> {}：{}").format(gateway_name, req.__str__()))

            return gateway.send_order(req)
        else:
            return ""

    def cancel_order(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel order request to a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("委托撤单 -> {}：{}").format(gateway_name, req))

            gateway.cancel_order(req)

    def send_quote(self, req: QuoteRequest, gateway_name: str) -> str:
        """
        Send new quote request to a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("报价下单 -> {}：{}").format(gateway_name, req))

            return gateway.send_quote(req)
        else:
            return ""

    def cancel_quote(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel quote request to a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("报价撤单 -> {}：{}").format(gateway_name, req))

            gateway.cancel_quote(req)

    def query_history(self, req: HistoryRequest, gateway_name: str) -> list[BarData]:
        """
        Query bar history data from a specific gateway.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            self.write_log(_("查询K线 -> {}：{}").format(gateway_name, req))

            return gateway.query_history(req)
        else:
            return []

    def close(self) -> None:
        """
        Make sure every gateway and app is closed properly before
        programme exit.
        """
        # Stop event engine first to prevent new timer event.
        self.event_engine.stop()

        for engine in self.engines.values():
            engine.close()

        for gateway in self.gateways.values():
            gateway.close()


class LogEngine(BaseEngine):
    """
    Provides log event output function.
    """

    level_map: dict[int, str] = {
        DEBUG: "DEBUG",
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
        CRITICAL: "CRITICAL",
    }

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, "log")

        self.active = SETTINGS["log.active"]

        self.register_log(EVENT_LOG)

    def process_log_event(self, event: Event) -> None:
        """Process log event"""
        if not self.active:
            return

        log: LogData = event.data
        level: str | int = self.level_map.get(log.level, log.level)
        logger.log(level, log.msg, gateway_name=log.gateway_name)

    def register_log(self, event_type: str) -> None:
        """Register log event handler"""
        self.event_engine.register(event_type, self.process_log_event)


class OmsEngine(BaseEngine):
    """
    Provides order management system function.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, "oms")

        self.ticks: dict[str, TickData] = {}
        self.orders: dict[str, OrderData] = {}
        self.trades: dict[str, TradeData] = {}
        self.positions: dict[str, PositionData] = {}
        self.accounts: dict[str, AccountData] = {}
        self.contracts: dict[str, ContractData] = {}
        self.quotes: dict[str, QuoteData] = {}

        self.active_orders: dict[str, OrderData] = {}
        self.active_quotes: dict[str, QuoteData] = {}

        self.offset_converters: dict[str, OffsetConverter] = {}

        self.mysql_module: object | None = None
        self.mysql_connection: object | None = None
        self.mysql_enabled: bool = False
        self.mysql_order_table: str = self._normalize_table_name(
            SETTINGS.get("mysql.order_table", "vt_order_data"),
            "vt_order_data"
        )
        self.mysql_trade_table: str = self._normalize_table_name(
            SETTINGS.get("mysql.trade_table", "vt_trade_data"),
            "vt_trade_data"
        )
        self.init_mysql_order_persistence()

        self.register_event()

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_QUOTE, self.process_quote_event)

    def _normalize_table_name(self, table_name: str, default: str) -> str:
        """Keep table name SQL-safe by allowing only letters, numbers and underscore."""
        filtered: str = "".join(c for c in str(table_name) if c.isalnum() or c == "_")
        return filtered or default

    def init_mysql_order_persistence(self) -> None:
        """Initialize mysql client and table schema for order/trade persistence."""
        enable: bool = bool(SETTINGS.get("mysql.order_persistence", True))
        database_name: str = str(SETTINGS.get("database.name", "")).lower()
        if not enable or database_name != "mysql":
            return

        try:
            import pymysql
        except ModuleNotFoundError:
            self.main_engine.write_log(
                _("找不到pymysql，已禁用订单和成交的MySQL持久化"),
                "OmsEngine"
            )
            return

        self.mysql_module = pymysql
        if not self._connect_mysql():
            return
        if not self._init_mysql_tables():
            return

        self.mysql_enabled = True
        self.main_engine.write_log(_("已启用订单和成交MySQL持久化"), "OmsEngine")

    def _connect_mysql(self) -> bool:
        """Create mysql connection by global database settings."""
        if not self.mysql_module:
            return False

        host: str = SETTINGS.get("database.host", "")
        port: int = int(SETTINGS.get("database.port", 0))
        user: str = SETTINGS.get("database.user", "")
        password: str = SETTINGS.get("database.password", "")
        database: str = SETTINGS.get("database.database", "")

        if not all([host, port, user, database]):
            self.main_engine.write_log(
                _("MySQL连接参数不完整，已禁用订单和成交持久化"),
                "OmsEngine"
            )
            return False

        try:
            self.mysql_connection = self.mysql_module.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                charset="utf8mb4",
                autocommit=True
            )
            return True
        except Exception:
            self.mysql_connection = None
            self.main_engine.write_log(
                _("连接MySQL失败，已禁用订单和成交持久化：{}").format(traceback.format_exc()),
                "OmsEngine"
            )
            return False

    def _execute_mysql(self, sql: str, params: tuple = ()) -> int:
        """Execute sql and reconnect once when mysql connection is broken."""
        if not self.mysql_connection and not self._connect_mysql():
            return -1

        error: str = ""
        for _ in range(2):
            try:
                with self.mysql_connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.rowcount
            except Exception:
                error = traceback.format_exc()
                self.mysql_connection = None
                if not self._connect_mysql():
                    break

        if error:
            self.main_engine.write_log(_("执行MySQL语句失败：{}").format(error), "OmsEngine")
        else:
            self.main_engine.write_log(_("执行MySQL语句失败，且重连数据库失败"), "OmsEngine")
        return -1

    def _order_exists_in_mysql(self, vt_orderid: str) -> bool:
        """Check whether order row exists in mysql."""
        if not self.mysql_enabled:
            return False

        sql: str = f"SELECT 1 FROM `{self.mysql_order_table}` WHERE `vt_orderid` = %s LIMIT 1;"
        return self._execute_mysql(sql, (vt_orderid,)) > 0

    def _init_mysql_tables(self) -> bool:
        """Create order/trade persistence tables if not exists."""
        order_sql: str = f"""
        CREATE TABLE IF NOT EXISTS `{self.mysql_order_table}` (
            `vt_orderid` VARCHAR(64) NOT NULL,
            `gateway_name` VARCHAR(32) NOT NULL,
            `orderid` VARCHAR(64) NOT NULL,
            `symbol` VARCHAR(32) NOT NULL,
            `exchange` VARCHAR(16) NOT NULL,
            `vt_symbol` VARCHAR(64) NOT NULL,
            `type` VARCHAR(32) NOT NULL,
            `direction` VARCHAR(16) DEFAULT NULL,
            `offset` VARCHAR(16) NOT NULL,
            `price` DOUBLE NOT NULL DEFAULT 0,
            `volume` DOUBLE NOT NULL DEFAULT 0,
            `traded` DOUBLE NOT NULL DEFAULT 0,
            `status` VARCHAR(16) NOT NULL,
            `status_msg` VARCHAR(255) NOT NULL DEFAULT '',
            `reference` VARCHAR(255) NOT NULL DEFAULT '',
            `order_time` DATETIME DEFAULT NULL,
            `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`vt_orderid`),
            KEY `idx_{self.mysql_order_table}_vt_symbol` (`vt_symbol`),
            KEY `idx_{self.mysql_order_table}_status` (`status`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        trade_sql: str = f"""
        CREATE TABLE IF NOT EXISTS `{self.mysql_trade_table}` (
            `vt_tradeid` VARCHAR(64) NOT NULL,
            `gateway_name` VARCHAR(32) NOT NULL,
            `tradeid` VARCHAR(64) NOT NULL,
            `orderid` VARCHAR(64) NOT NULL,
            `vt_orderid` VARCHAR(64) NOT NULL,
            `symbol` VARCHAR(32) NOT NULL,
            `exchange` VARCHAR(16) NOT NULL,
            `vt_symbol` VARCHAR(64) NOT NULL,
            `direction` VARCHAR(16) DEFAULT NULL,
            `offset` VARCHAR(16) NOT NULL,
            `price` DOUBLE NOT NULL DEFAULT 0,
            `volume` DOUBLE NOT NULL DEFAULT 0,
            `trade_time` DATETIME DEFAULT NULL,
            `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (`vt_tradeid`),
            KEY `idx_{self.mysql_trade_table}_vt_orderid` (`vt_orderid`),
            KEY `idx_{self.mysql_trade_table}_vt_symbol` (`vt_symbol`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        if self._execute_mysql(order_sql) < 0:
            return False
        if self._execute_mysql(trade_sql) < 0:
            return False
        return True

    def _save_order_to_mysql(self, order: OrderData) -> None:
        """Upsert latest order snapshot to mysql."""
        if not self.mysql_enabled:
            return

        order_time = order.datetime.replace(tzinfo=None) if order.datetime else None
        sql: str = f"""
        INSERT INTO `{self.mysql_order_table}` (
            `vt_orderid`, `gateway_name`, `orderid`, `symbol`, `exchange`, `vt_symbol`,
            `type`, `direction`, `offset`, `price`, `volume`, `traded`, `status`,
            `status_msg`, `reference`, `order_time`
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            `gateway_name` = VALUES(`gateway_name`),
            `orderid` = VALUES(`orderid`),
            `symbol` = VALUES(`symbol`),
            `exchange` = VALUES(`exchange`),
            `vt_symbol` = VALUES(`vt_symbol`),
            `type` = VALUES(`type`),
            `direction` = VALUES(`direction`),
            `offset` = VALUES(`offset`),
            `price` = VALUES(`price`),
            `volume` = VALUES(`volume`),
            `traded` = VALUES(`traded`),
            `status` = VALUES(`status`),
            `status_msg` = VALUES(`status_msg`),
            `reference` = VALUES(`reference`),
            `order_time` = VALUES(`order_time`);
        """
        params: tuple = (
            order.vt_orderid,
            order.gateway_name,
            order.orderid,
            order.symbol,
            order.exchange.value,
            order.vt_symbol,
            order.type.value,
            order.direction.value if order.direction else "",
            order.offset.value,
            order.price,
            order.volume,
            order.traded,
            order.status.value,
            order.status_msg,
            order.reference,
            order_time
        )
        self._execute_mysql(sql, params)

    def _save_trade_to_mysql(self, trade: TradeData) -> None:
        """Insert trade and update corresponding order traded/status."""
        if not self.mysql_enabled:
            return

        trade_time = trade.datetime.replace(tzinfo=None) if trade.datetime else None
        trade_sql: str = f"""
        INSERT IGNORE INTO `{self.mysql_trade_table}` (
            `vt_tradeid`, `gateway_name`, `tradeid`, `orderid`, `vt_orderid`, `symbol`,
            `exchange`, `vt_symbol`, `direction`, `offset`, `price`, `volume`, `trade_time`
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        trade_params: tuple = (
            trade.vt_tradeid,
            trade.gateway_name,
            trade.tradeid,
            trade.orderid,
            trade.vt_orderid,
            trade.symbol,
            trade.exchange.value,
            trade.vt_symbol,
            trade.direction.value if trade.direction else "",
            trade.offset.value,
            trade.price,
            trade.volume,
            trade_time
        )
        rowcount: int = self._execute_mysql(trade_sql, trade_params)
        if rowcount <= 0:
            return

        self._update_order_by_trade(trade)

    def _update_order_by_trade(self, trade: TradeData) -> None:
        """Backfill order traded/status whenever a new trade is inserted."""
        if not self.mysql_enabled:
            return

        trade_time = trade.datetime.replace(tzinfo=None) if trade.datetime else None
        part_status: str = Status.PARTTRADED.value
        all_status: str = Status.ALLTRADED.value
        update_sql: str = f"""
        UPDATE `{self.mysql_order_table}`
        SET
            `traded` = LEAST(`volume`, `traded` + %s),
            `status` = CASE
                WHEN `volume` > 0 AND (`traded` + %s) >= `volume` THEN %s
                WHEN (`traded` + %s) > 0 THEN %s
                ELSE `status`
            END
        WHERE `vt_orderid` = %s;
        """
        params: tuple = (trade.volume, trade.volume, all_status, trade.volume, part_status, trade.vt_orderid)

        update_result: int = self._execute_mysql(update_sql, params)
        if update_result < 0:
            return
        if self._order_exists_in_mysql(trade.vt_orderid):
            return

        # Trade may arrive before order snapshot, write order first and retry once.
        order: OrderData | None = self.get_order(trade.vt_orderid)
        if order:
            self._save_order_to_mysql(order)
            self._execute_mysql(update_sql, params)
            return

        fallback_sql: str = f"""
        INSERT INTO `{self.mysql_order_table}` (
            `vt_orderid`, `gateway_name`, `orderid`, `symbol`, `exchange`, `vt_symbol`,
            `type`, `direction`, `offset`, `price`, `volume`, `traded`, `status`,
            `status_msg`, `reference`, `order_time`
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '', '', %s)
        ON DUPLICATE KEY UPDATE
            `traded` = LEAST(`volume`, `traded` + VALUES(`traded`)),
            `status` = CASE
                WHEN `volume` > 0 AND (`traded` + VALUES(`traded`)) >= `volume` THEN %s
                WHEN (`traded` + VALUES(`traded`)) > 0 THEN %s
                ELSE `status`
            END;
        """
        fallback_params: tuple = (
            trade.vt_orderid,
            trade.gateway_name,
            trade.orderid,
            trade.symbol,
            trade.exchange.value,
            trade.vt_symbol,
            "",
            trade.direction.value if trade.direction else "",
            trade.offset.value,
            trade.price,
            trade.volume,
            trade.volume,
            part_status,
            trade_time,
            all_status,
            part_status
        )
        self._execute_mysql(fallback_sql, fallback_params)

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data
        self.ticks[tick.vt_symbol] = tick

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data
        self.orders[order.vt_orderid] = order

        # If order is active, then update data in dict.
        if order.is_active():
            self.active_orders[order.vt_orderid] = order
        # Otherwise, pop inactive order from in dict
        elif order.vt_orderid in self.active_orders:
            self.active_orders.pop(order.vt_orderid)

        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(order.gateway_name, None)
        if converter:
            converter.update_order(order)

        self._save_order_to_mysql(order)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        self.trades[trade.vt_tradeid] = trade

        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(trade.gateway_name, None)
        if converter:
            converter.update_trade(trade)

        self._save_trade_to_mysql(trade)

    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data
        self.positions[position.vt_positionid] = position

        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(position.gateway_name, None)
        if converter:
            converter.update_position(position)

    def process_account_event(self, event: Event) -> None:
        """"""
        account: AccountData = event.data
        self.accounts[account.vt_accountid] = account

    def process_contract_event(self, event: Event) -> None:
        """"""
        contract: ContractData = event.data
        self.contracts[contract.vt_symbol] = contract

        # Initialize offset converter for each gateway
        if contract.gateway_name not in self.offset_converters:
            self.offset_converters[contract.gateway_name] = OffsetConverter(self)

    def process_quote_event(self, event: Event) -> None:
        """"""
        quote: QuoteData = event.data
        self.quotes[quote.vt_quoteid] = quote

        # If quote is active, then update data in dict.
        if quote.is_active():
            self.active_quotes[quote.vt_quoteid] = quote
        # Otherwise, pop inactive quote from in dict
        elif quote.vt_quoteid in self.active_quotes:
            self.active_quotes.pop(quote.vt_quoteid)

    def get_tick(self, vt_symbol: str) -> TickData | None:
        """
        Get latest market tick data by vt_symbol.
        """
        return self.ticks.get(vt_symbol, None)

    def get_order(self, vt_orderid: str) -> OrderData | None:
        """
        Get latest order data by vt_orderid.
        """
        return self.orders.get(vt_orderid, None)

    def get_trade(self, vt_tradeid: str) -> TradeData | None:
        """
        Get trade data by vt_tradeid.
        """
        return self.trades.get(vt_tradeid, None)

    def get_position(self, vt_positionid: str) -> PositionData | None:
        """
        Get latest position data by vt_positionid.
        """
        return self.positions.get(vt_positionid, None)

    def get_account(self, vt_accountid: str) -> AccountData | None:
        """
        Get latest account data by vt_accountid.
        """
        return self.accounts.get(vt_accountid, None)

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        """
        Get contract data by vt_symbol.
        """
        return self.contracts.get(vt_symbol, None)

    def get_quote(self, vt_quoteid: str) -> QuoteData | None:
        """
        Get latest quote data by vt_orderid.
        """
        return self.quotes.get(vt_quoteid, None)

    def get_all_ticks(self) -> list[TickData]:
        """
        Get all tick data.
        """
        return list(self.ticks.values())

    def get_all_orders(self) -> list[OrderData]:
        """
        Get all order data.
        """
        return list(self.orders.values())

    def get_all_trades(self) -> list[TradeData]:
        """
        Get all trade data.
        """
        return list(self.trades.values())

    def get_all_positions(self) -> list[PositionData]:
        """
        Get all position data.
        """
        return list(self.positions.values())

    def get_all_accounts(self) -> list[AccountData]:
        """
        Get all account data.
        """
        return list(self.accounts.values())

    def get_all_contracts(self) -> list[ContractData]:
        """
        Get all contract data.
        """
        return list(self.contracts.values())

    def get_all_quotes(self) -> list[QuoteData]:
        """
        Get all quote data.
        """
        return list(self.quotes.values())

    def get_all_active_orders(self) -> list[OrderData]:
        """
        Get all active orders.
        """
        return list(self.active_orders.values())

    def get_all_active_quotes(self) -> list[QuoteData]:
        """
        Get all active quotes.
        """
        return list(self.active_quotes.values())

    def update_order_request(self, req: OrderRequest, vt_orderid: str, gateway_name: str) -> None:
        """
        Update order request to offset converter.
        """
        converter: OffsetConverter | None = self.offset_converters.get(gateway_name, None)
        if converter:
            converter.update_order_request(req, vt_orderid)

    def convert_order_request(
        self,
        req: OrderRequest,
        gateway_name: str,
        lock: bool,
        net: bool = False
    ) -> list[OrderRequest]:
        """
        Convert original order request according to given mode.
        """
        converter: OffsetConverter | None = self.offset_converters.get(gateway_name, None)
        if not converter:
            return [req]

        reqs: list[OrderRequest] = converter.convert_order_request(req, lock, net)
        return reqs

    def get_converter(self, gateway_name: str) -> OffsetConverter | None:
        """
        Get offset converter object of specific gateway.
        """
        return self.offset_converters.get(gateway_name, None)

    def close(self) -> None:
        """Close mysql connection on engine shutdown."""
        if not self.mysql_connection:
            return

        try:
            self.mysql_connection.close()
        except Exception:
            self.main_engine.write_log(
                _("关闭MySQL连接失败：{}").format(traceback.format_exc()),
                "OmsEngine"
            )
        finally:
            self.mysql_connection = None
            self.mysql_enabled = False


class EmailEngine(BaseEngine):
    """
    Provides email sending function.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, "email")

        self.thread: Thread = Thread(target=self.run)
        self.queue: Queue = Queue()
        self.active: bool = False

    def send_email(self, subject: str, content: str, receiver: str | None = None) -> None:
        """"""
        # Start email engine when sending first email.
        if not self.active:
            self.start()

        # Use default receiver if not specified.
        if not receiver:
            receiver = SETTINGS["email.receiver"]

        msg: EmailMessage = EmailMessage()
        msg["From"] = SETTINGS["email.sender"]
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.set_content(content)

        self.queue.put(msg)

    def run(self) -> None:
        """"""
        server: str = SETTINGS["email.server"]
        port: int = SETTINGS["email.port"]
        username: str = SETTINGS["email.username"]
        password: str = SETTINGS["email.password"]

        while self.active:
            try:
                msg: EmailMessage = self.queue.get(block=True, timeout=1)

                try:
                    with smtplib.SMTP_SSL(server, port) as smtp:
                        smtp.login(username, password)
                        smtp.send_message(msg)
                        smtp.close()
                except Exception:
                    log_msg: str = _("邮件发送失败: {}").format(traceback.format_exc())
                    self.main_engine.write_log(log_msg, "EmailEngine")
            except Empty:
                pass

    def start(self) -> None:
        """"""
        self.active = True
        self.thread.start()

    def close(self) -> None:
        """"""
        if not self.active:
            return

        self.active = False
        self.thread.join()
