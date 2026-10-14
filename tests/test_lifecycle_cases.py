"""Public-interface integration cases copied into a disposable upstream tree."""
from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Direction, Exchange, Offset, Product, Status
from vnpy.trader.engine import MainEngine
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    BarData, CancelRequest, ContractData, OrderData, OrderRequest,
    SubscribeRequest, TickData, TradeData,
)
from vnpy.trader.ui import QtCore, QtGui, QtWidgets
from vnpy.trader.utility import TEMP_DIR
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_LOG, EVENT_PORTFOLIO_STRATEGY
from vnpy_portfoliostrategy.engine import StrategyEngine
from vnpy_portfoliostrategy.template import StrategyTemplate
from vnpy_portfoliostrategy.ui.widget import PortfolioStrategyManager

SYMBOL = "SYNTHETIC.SSE"
TRACES: dict[str, Any] = {}


class OfflineGateway(BaseGateway):
    """Only this explicit local boundary substitutes for an external broker."""

    default_name = "OFFLINE_START_TEST"
    exchanges = [Exchange.SSE]

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        super().__init__(event_engine, gateway_name)
        self.requests: list[OrderRequest] = []
        self.cancellations: list[str] = []
        self.subscriptions: list[str] = []
        self.orders: dict[str, OrderData] = {}
        self.defer_cancellation = False
        self.last_trade: TradeData | None = None

    def connect(self, setting: dict) -> None:
        raise AssertionError("External connections are forbidden in this fixture")

    def close(self) -> None:
        pass

    def subscribe(self, req: SubscribeRequest) -> None:
        self.subscriptions.append(req.vt_symbol)

    def query_account(self) -> None:
        raise AssertionError("No real account in this fixture")

    def query_position(self) -> None:
        raise AssertionError("No real position query in this fixture")

    def send_order(self, req: OrderRequest) -> str:
        self.requests.append(req)
        order = replace(req.create_order_data(str(len(self.requests)), self.gateway_name), status=Status.NOTTRADED)
        self.orders[order.orderid] = order
        # Tests deliver the callback explicitly, including deliberately late responses.
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        self.cancellations.append(req.orderid)
        if self.defer_cancellation:
            return
        order = replace(self.orders[req.orderid], status=Status.CANCELLED)
        self.orders[req.orderid] = order
        self.on_order(order)

    def partial_fill(self, orderid: str, tradeid: str, volume: int) -> None:
        order = self.orders[orderid]
        partial = replace(order, traded=order.traded + volume, status=Status.PARTTRADED)
        self.orders[orderid] = partial
        self.on_order(partial)
        trade = TradeData(
            symbol=order.symbol, exchange=order.exchange, orderid=order.orderid,
            tradeid=tradeid, direction=Direction.LONG, offset=Offset.OPEN,
            price=order.price, volume=volume, gateway_name=self.gateway_name,
        )
        self.last_trade = trade
        self.on_trade(trade)


class EngineeringStrategy(StrategyTemplate):
    author = "Offline regression fixture"
    parameters = ["order_on_tick", "mutate_start"]
    variables = ["init_calls", "start_calls", "stop_calls", "tick_calls"]
    order_on_tick = False
    mutate_start = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_calls = self.start_calls = self.stop_calls = self.tick_calls = 0
        self.start_exception: Exception | None = None
        self.init_exception: Exception | None = None
        self.stop_exception: Exception | None = None
        self.tick_exception: Exception | None = None
        self.start_result: Any = None
        self.callback_states: list[tuple[bool, bool]] = []
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.start_release.set()
        self.nested_failure = False
        super().__init__(*args, **kwargs)

    def on_init(self) -> None:
        self.init_calls += 1
        if self.init_exception is not None:
            raise self.init_exception

    def on_start(self) -> None:
        self.start_calls += 1
        self.callback_states.append((self.inited, self.trading))
        assert self.buy(SYMBOL, 4.0, 100) == [], "Trading must stay disabled during startup"
        self.start_entered.set()
        if not self.start_release.wait(4):
            raise TimeoutError("The test failed to release the startup callback")
        if self.mutate_start:
            self.set_target(SYMBOL, 200)
        if self.nested_failure:
            def failed_inner_callback() -> None:
                raise RuntimeError("inner callback failure")
            self.strategy_engine.call_strategy_func(self, failed_inner_callback)
        if self.start_exception is not None:
            raise self.start_exception
        return self.start_result

    def on_stop(self) -> None:
        self.stop_calls += 1
        if self.stop_exception is not None:
            raise self.stop_exception

    def on_tick(self, tick: TickData) -> None:
        self.tick_calls += 1
        if self.tick_exception is not None:
            raise self.tick_exception
        if self.order_on_tick:
            self.buy(tick.vt_symbol, tick.last_price, 200)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        pass


@dataclass
class Rig:
    main: MainEngine
    engine: StrategyEngine
    gateway: OfflineGateway
    ui: PortfolioStrategyManager
    app: QtWidgets.QApplication
    events: list[dict]
    logs: list[str]

    def flush(self) -> None:
        # Two FIFO barriers also collect events queued by the first set of callbacks.
        for _ in range(2):
            done = threading.Event()
            marker = "portfolioRegressionBarrier"

            def arrived(event: Event, completion: threading.Event = done) -> None:
                if event.data is completion:
                    completion.set()

            self.main.event_engine.register(marker, arrived)
            self.main.event_engine.put(Event(marker, done))
            assert done.wait(4), "The native event thread did not deliver the barrier"
            self.main.event_engine.unregister(marker, arrived)
            self.app.processEvents()

    def add(self, name: str = "engineering", **setting: Any) -> EngineeringStrategy:
        self.engine.add_strategy("EngineeringStrategy", name, [SYMBOL], setting)
        self.flush()
        strategy = self.engine.strategies[name]
        assert isinstance(strategy, EngineeringStrategy)
        assert name in self.ui.managers
        return strategy

    def initialize(self, strategy: EngineeringStrategy) -> None:
        assert self.engine.init_strategy(strategy.strategy_name) is None
        self.engine.init_executor.submit(lambda: None).result(timeout=4)
        self.flush()

    def tick(self) -> None:
        self.gateway.on_tick(TickData(
            symbol="SYNTHETIC", exchange=Exchange.SSE,
            datetime=datetime(2026, 10, 14, 10, 0), last_price=4.0,
            gateway_name=self.gateway.gateway_name,
        ))
        self.flush()

    def buttons(self, strategy: EngineeringStrategy) -> list[bool]:
        widget = self.ui.managers[strategy.strategy_name]
        return [button.isEnabled() for button in [
            widget.init_button, widget.start_button, widget.stop_button,
            widget.edit_button, widget.remove_button,
        ]]

    def states(self, strategy: EngineeringStrategy) -> list[list[bool]]:
        return [
            [item["variables"]["inited"], item["variables"]["trading"]]
            for item in self.events if item["strategy_name"] == strategy.strategy_name
        ]


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QtWidgets.QApplication]:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        QtGui.QFontDatabase.addApplicationFont(str(font))
    yield app
    app.processEvents()
    trace = os.environ.get("PORTFOLIO_TEST_TRACE_PATH")
    if trace:
        Path(trace).write_text(json.dumps(TRACES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def rig(qt_app: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    previous = Path.cwd()
    # Lifecycle tests do not query historical bars; substitute this unused service.
    # Real SQLite behavior is verified separately by the foundation suite.
    monkeypatch.setattr("vnpy_portfoliostrategy.engine.get_database", lambda: object())
    for name in [StrategyEngine.setting_filename, StrategyEngine.data_filename]:
        (TEMP_DIR / name).unlink(missing_ok=True)
    event_engine = EventEngine(interval=0.05)
    main = MainEngine(event_engine)
    engine = main.add_app(PortfolioStrategyApp)
    gateway = main.add_gateway(OfflineGateway)
    assert isinstance(engine, StrategyEngine)
    assert isinstance(gateway, OfflineGateway)
    ui = PortfolioStrategyManager(main, event_engine)
    engine.classes["EngineeringStrategy"] = EngineeringStrategy
    events: list[dict] = []
    logs: list[str] = []

    def collect_state(event: Event) -> None:
        events.append(deepcopy(event.data))

    def collect_log(event: Event) -> None:
        logs.append(event.data.msg)

    event_engine.register(EVENT_PORTFOLIO_STRATEGY, collect_state)
    event_engine.register(EVENT_PORTFOLIO_LOG, collect_log)
    built = Rig(main, engine, gateway, ui, qt_app, events, logs)
    gateway.on_contract(ContractData(
        symbol="SYNTHETIC", exchange=Exchange.SSE, name="Offline fixture",
        product=Product.EQUITY, size=1, pricetick=0.001, min_volume=100,
        gateway_name=gateway.gateway_name,
    ))
    built.flush()
    assert main.get_contract(SYMBOL) is not None
    try:
        yield built
    finally:
        for strategy in engine.strategies.values():
            if isinstance(strategy, EngineeringStrategy):
                strategy.start_release.set()
        engine.init_executor.shutdown(wait=True, cancel_futures=True)
        main.close()
        ui.close()
        ui.deleteLater()
        qt_app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        qt_app.processEvents()
        os.chdir(previous)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError])
def test_failed_start_reports_stopped_state_and_blocks_orders(rig: Rig, error_type: type[Exception]) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_exception = error_type("intentional startup failure")
    assert rig.engine.start_strategy(strategy.strategy_name) is None
    rig.flush()
    TRACES["failed_" + error_type.__name__] = {
        "states": rig.states(strategy), "buttons": rig.buttons(strategy),
        "callback_states": strategy.callback_states,
        "start_calls": strategy.start_calls,
        "order_count": len(rig.gateway.requests),
    }
    assert strategy.inited is False and strategy.trading is False
    assert rig.states(strategy)[-1] == [False, False]
    assert [False, True] not in rig.states(strategy)
    assert any(error_type.__name__ + ": intentional startup failure" in line for line in rig.logs)
    assert strategy.buy(SYMBOL, 4, 100) == []
    assert rig.gateway.requests == []
    rig.tick()
    assert strategy.tick_calls == 0


@pytest.mark.parametrize("return_value", [None, False, 0, "complete"])
def test_success_preserves_callback_order_and_none_return(rig: Rig, return_value: Any) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_result = return_value
    assert rig.engine.start_strategy(strategy.strategy_name) is None
    rig.flush()
    assert strategy.callback_states == [(True, False)]
    assert strategy.inited is True and strategy.trading is True
    assert rig.states(strategy) == [[False, False], [True, False], [True, True]]
    assert rig.buttons(strategy) == [False, False, True, False, False]
    assert rig.gateway.requests == []
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is True and strategy.trading is False
    assert strategy.stop_calls == 1
    assert rig.buttons(strategy) == [False, True, False, True, True]


def test_failed_start_requires_explicit_reinitialization(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_exception = RuntimeError("failure once")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    strategy.start_exception = None
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.start_calls == 1
    assert strategy.trading is False
    rig.initialize(strategy)
    assert strategy.init_calls == 2
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.start_calls == 2 and strategy.trading is True
    assert strategy.callback_states == [(True, False), (True, False)]


def test_failure_retains_targets_positions_and_late_fill_evidence(rig: Rig) -> None:
    strategy = rig.add(mutate_start=True)
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 300)[0]
    orderid = identity.split(".")[-1]
    rig.gateway.on_order(rig.gateway.orders[orderid])
    rig.flush()
    rig.gateway.partial_fill(orderid, "first", 100)
    rig.flush()
    assert strategy.get_pos(SYMBOL) == 100
    rig.gateway.defer_cancellation = True
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert rig.main.get_order(identity).traded == 100
    assert identity in strategy.active_orderids
    strategy.start_exception = RuntimeError("target mutated before failure")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    rig.gateway.partial_fill(orderid, "late", 100)
    assert rig.gateway.last_trade is not None
    rig.gateway.on_trade(replace(rig.gateway.last_trade))
    rig.flush()
    assert strategy.get_pos(SYMBOL) == 200
    assert strategy.get_target(SYMBOL) == 200
    assert rig.engine.orderid_strategy_map[identity] is strategy
    assert identity in strategy.active_orderids
    assert rig.main.get_order(identity).traded == 200
    assert len(rig.gateway.requests) == 1
    # Explicit engine cancellation remains available even while the strategy is stopped.
    rig.gateway.defer_cancellation = False
    rig.engine.cancel_all(strategy)
    rig.flush()
    assert not strategy.active_orderids
    assert rig.main.get_order(identity).traded == 200
    assert strategy.get_pos(SYMBOL) == 200
    assert strategy.inited is False and strategy.trading is False
    TRACES["retained_evidence"] = {
        "position": strategy.get_pos(SYMBOL), "target": strategy.get_target(SYMBOL),
        "traded": rig.main.get_order(identity).traded, "requests": len(rig.gateway.requests),
        "cancellations": rig.gateway.cancellations,
    }


def test_stopping_after_failed_restart_cancels_pending_order(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 300)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.flush()
    rig.gateway.defer_cancellation = True
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert identity in strategy.active_orderids
    rig.gateway.defer_cancellation = False
    strategy.mutate_start = True
    strategy.start_exception = RuntimeError("restart failed with a pending order")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    rig.gateway.partial_fill(order.orderid, "late-before-stop", 100)
    rig.flush()
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    TRACES["stop_after_failed_restart"] = {
        "cancellations": len(rig.gateway.cancellations),
        "active_orders": len(strategy.active_orderids),
        "stop_calls": strategy.stop_calls,
    }
    assert rig.main.get_order(identity).status == Status.CANCELLED
    assert not strategy.active_orderids
    assert strategy.stop_calls == 1
    persisted = json.loads((TEMP_DIR / rig.engine.data_filename).read_text(encoding="utf-8"))
    assert persisted[strategy.strategy_name]["pos_data"].get(SYMBOL) == 100
    assert persisted[strategy.strategy_name]["target_data"].get(SYMBOL) == 200
    assert len(rig.states(strategy)) >= 6
    assert rig.states(strategy)[-1] == [False, False]
    TRACES["stop_after_failed_restart"]["persisted_position"] = 100
    TRACES["stop_after_failed_restart"]["persisted_target"] = 200


def test_failed_restart_cannot_remove_pending_order_evidence(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 300)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.flush()
    rig.gateway.defer_cancellation = True
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    strategy.start_exception = RuntimeError("restart failed before removal")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    removed = rig.engine.remove_strategy(strategy.strategy_name)
    rig.flush()
    TRACES["removal_after_failed_restart"] = {
        "removed": removed, "route_retained": identity in rig.engine.orderid_strategy_map,
        "strategy_retained": strategy.strategy_name in rig.engine.strategies,
    }
    assert removed is False
    assert rig.engine.orderid_strategy_map[identity] is strategy
    rig.gateway.partial_fill(order.orderid, "after-rejected-removal", 100)
    rig.flush()
    assert strategy.get_pos(SYMBOL) == 100


def test_callback_failure_on_event_thread_during_start_cannot_reenable_trading(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_release.clear()
    worker = threading.Thread(target=rig.engine.start_strategy, args=(strategy.strategy_name,))
    worker.start()
    try:
        assert strategy.start_entered.wait(4)
        strategy.tick_exception = RuntimeError("tick failed during startup")
        rig.tick()
    finally:
        strategy.start_release.set()
        worker.join(4)
    assert not worker.is_alive()
    rig.flush()
    TRACES["interleaved_callback_failure"] = {
        "states": rig.states(strategy), "start_calls": strategy.start_calls,
        "tick_calls": strategy.tick_calls, "callback_states": strategy.callback_states,
    }
    assert strategy.tick_calls == 1
    assert strategy.inited is False and strategy.trading is False
    assert rig.states(strategy)[-1] == [False, False]


def test_nested_callback_failure_cannot_reenable_trading(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.nested_failure = True
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is False and strategy.trading is False
    assert any("inner callback failure" in line for line in rig.logs)


def test_healthy_public_lifecycle_trace(rig: Rig) -> None:
    strategy = rig.add(order_on_tick=True)
    initial_buttons = rig.buttons(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.start_calls == 0
    widget = rig.ui.managers[strategy.strategy_name]
    widget.init_button.click()
    rig.engine.init_executor.submit(lambda: None).result(timeout=4)
    rig.flush()
    rig.initialize(strategy)
    assert strategy.init_calls == 1
    widget.start_button.click()
    rig.flush()
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.start_calls == 1
    rig.tick()
    assert len(rig.gateway.requests) == 1
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.gateway.partial_fill(order.orderid, "normal", 100)
    rig.flush()
    widget.stop_button.click()
    rig.flush()
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.stop_calls == 1
    assert strategy.get_pos(SYMBOL) == 100
    assert rig.main.get_order(order.vt_orderid).traded == 100
    assert not strategy.active_orderids
    persisted = json.loads((TEMP_DIR / rig.engine.data_filename).read_text(encoding="utf-8"))
    assert "inited" not in persisted[strategy.strategy_name]
    assert "trading" not in persisted[strategy.strategy_name]
    TRACES["healthy_lifecycle"] = {
        "states": rig.states(strategy), "initial_buttons": initial_buttons,
        "final_buttons": rig.buttons(strategy),
        "callbacks": [strategy.init_calls, strategy.start_calls, strategy.stop_calls, strategy.tick_calls],
        "callback_states": strategy.callback_states,
        "subscriptions": rig.gateway.subscriptions,
        "requests": [
            [req.symbol, req.price, req.volume, req.direction.value, req.offset.value, req.reference]
            for req in rig.gateway.requests
        ],
        "cancellations": rig.gateway.cancellations, "position": strategy.get_pos(SYMBOL),
        "persisted": persisted,
    }


def test_all_strategy_operations_isolate_one_failure(rig: Rig) -> None:
    healthy = rig.add("healthy")
    failed = rig.add("failed")
    failed.start_exception = RuntimeError("only this strategy fails")
    rig.engine.init_all_strategies()
    rig.engine.init_executor.submit(lambda: None).result(timeout=4)
    rig.flush()
    rig.engine.start_all_strategies()
    rig.flush()
    assert healthy.inited is True and healthy.trading is True
    assert failed.inited is False and failed.trading is False
    rig.engine.stop_all_strategies()
    rig.flush()
    assert healthy.stop_calls == 1 and healthy.trading is False
    assert failed.stop_calls == 0


def test_stop_callback_exception_still_cancels_and_persists(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 100)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.flush()
    strategy.stop_exception = RuntimeError("stop callback failed")
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is False and strategy.trading is False
    assert rig.main.get_order(identity).status == Status.CANCELLED
    assert not strategy.active_orderids
    assert (TEMP_DIR / rig.engine.data_filename).is_file()
    assert any("stop callback failed" in line for line in rig.logs)


def test_bounded_lifecycle_campaign(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    repetitions = int(os.environ.get("PORTFOLIO_TEST_REPETITIONS", "50"))
    for _ in range(repetitions):
        rig.engine.start_strategy(strategy.strategy_name)
        rig.engine.start_strategy(strategy.strategy_name)
        rig.flush()
        assert strategy.inited is True and strategy.trading is True
        rig.engine.stop_strategy(strategy.strategy_name)
        rig.engine.stop_strategy(strategy.strategy_name)
        rig.flush()
        assert strategy.inited is True and strategy.trading is False
    assert strategy.start_calls == repetitions and strategy.stop_calls == repetitions
    assert len(rig.states(strategy)) == repetitions * 2 + 2
    assert rig.gateway.requests == []
    TRACES["lifecycle_campaign"] = {
        "repetitions": repetitions, "start_calls": strategy.start_calls,
        "stop_calls": strategy.stop_calls, "state_events": len(rig.states(strategy)),
    }


def test_bounded_failure_and_manual_recovery_campaign(rig: Rig) -> None:
    strategy = rig.add()
    strategy.start_exception = RuntimeError("repeated controlled failure")
    repetitions = int(os.environ.get("PORTFOLIO_TEST_REPETITIONS", "50"))
    for _ in range(repetitions):
        rig.initialize(strategy)
        rig.engine.start_strategy(strategy.strategy_name)
        rig.flush()
        assert strategy.inited is False and strategy.trading is False
        rig.engine.start_strategy(strategy.strategy_name)
        rig.flush()
    assert strategy.init_calls == repetitions and strategy.start_calls == repetitions
    assert [False, True] not in rig.states(strategy)
    assert rig.gateway.requests == []
    strategy.start_exception = None
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is True and strategy.trading is True
    TRACES["failure_recovery_campaign"] = {
        "repetitions": repetitions, "init_calls": strategy.init_calls,
        "start_calls": strategy.start_calls, "state_events": len(rig.states(strategy)),
        "gateway_requests": len(rig.gateway.requests),
    }


def test_process_exit_from_callback_is_not_swallowed(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    rig.initialize(strategy)

    def exit_requested() -> None:
        raise SystemExit("intentional process exit")

    monkeypatch.setattr(strategy, "on_start", exit_requested)
    with pytest.raises(SystemExit, match="intentional process exit"):
        rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is True and strategy.trading is False
    assert rig.states(strategy)[-1] == [True, False]


def test_idle_strategy_can_still_be_edited_and_removed(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    rig.engine.edit_strategy(strategy.strategy_name, {"mutate_start": True})
    rig.flush()
    assert strategy.mutate_start is True
    assert rig.engine.remove_strategy(strategy.strategy_name) is True
    assert strategy.strategy_name not in rig.engine.strategies
    assert strategy not in rig.engine.symbol_strategy_map[SYMBOL]


def test_closing_uninitialized_strategy_preserves_saved_positions(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 100)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.gateway.partial_fill(order.orderid, "saved", 100)
    rig.flush()
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert rig.main.get_order(identity).traded == 100
    path = TEMP_DIR / rig.engine.data_filename
    before = path.read_bytes()
    reopened = StrategyEngine(rig.main, rig.main.event_engine)
    try:
        reopened.classes["EngineeringStrategy"] = EngineeringStrategy
        reopened.load_strategy_data()
        reopened.load_strategy_setting()
        loaded = reopened.strategies[strategy.strategy_name]
        assert loaded.inited is False and loaded.get_pos(SYMBOL) == 0
        reopened.stop_strategy(loaded.strategy_name)
        reopened.close()
        assert path.read_bytes() == before, "Never overwrite saved positions with uninitialized defaults"
    finally:
        reopened.init_executor.shutdown(wait=True, cancel_futures=True)


def test_reusing_removed_name_does_not_apply_old_failure_cleanup(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_exception = RuntimeError("failed instance will be removed")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert rig.engine.remove_strategy(strategy.strategy_name) is True
    fresh = rig.add(strategy.strategy_name)
    before = json.loads((TEMP_DIR / rig.engine.data_filename).read_text(encoding="utf-8"))
    assert before == {}
    rig.engine.stop_strategy(fresh.strategy_name)
    rig.flush()
    assert json.loads((TEMP_DIR / rig.engine.data_filename).read_text(encoding="utf-8")) == before
