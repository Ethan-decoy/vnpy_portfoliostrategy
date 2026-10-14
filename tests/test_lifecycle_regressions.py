"""Lifecycle recovery regressions through the native engine and Qt interface."""
import json
import threading
from dataclasses import replace
from copy import deepcopy

import pytest

from vnpy.trader.ui import QtCore
from vnpy.event import Event
from vnpy.trader.constant import Status
from vnpy_portfoliostrategy.base import EVENT_PORTFOLIO_STRATEGY
from vnpy.trader.utility import TEMP_DIR, save_json
from vnpy_portfoliostrategy.ui.widget import StrategyManager
from test_lifecycle_cases import EngineeringStrategy, Rig, SYMBOL

pytest_plugins = ["test_lifecycle_cases"]


def test_stopping_fresh_ready_strategy_has_no_side_effects(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    path = TEMP_DIR / rig.engine.data_filename
    before = path.read_bytes()
    events = len(rig.events)
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert len(rig.events) == events
    assert path.read_bytes() == before


def test_stopping_reinitialized_ready_strategy_has_no_side_effects(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_exception = RuntimeError("recoverable startup error")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    strategy.start_exception = None
    rig.initialize(strategy)
    assert strategy.inited is True and strategy.trading is False
    path = TEMP_DIR / rig.engine.data_filename
    before = path.read_bytes()
    events = len(rig.events)
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert len(rig.events) == events, "Successful reinitialization must clear old failure cleanup"
    assert path.read_bytes() == before


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError])
def test_initialization_failure_stays_uninitialized_and_preserves_saved_data(rig: Rig, error: type[Exception]) -> None:
    strategy = rig.add()
    rig.engine.strategy_data[strategy.strategy_name] = {"pos_data": {SYMBOL: 500}, "target_data": {SYMBOL: 600}}
    path = TEMP_DIR / rig.engine.data_filename
    save_json(rig.engine.data_filename, rig.engine.strategy_data)
    before = path.read_bytes()
    strategy.init_exception = error("failed before restoring saved variables")
    rig.initialize(strategy)
    assert strategy.inited is False and strategy.trading is False
    assert rig.states(strategy)[-1] == [False, False]
    assert rig.gateway.subscriptions == []
    assert strategy.get_pos(SYMBOL) == 0
    assert rig.buttons(strategy) == [True, False, False, True, True]
    rig.engine.start_strategy(strategy.strategy_name)
    assert strategy.start_calls == 0
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.engine.close()
    assert path.read_bytes() == before, "Failed initialization must not overwrite unrecovered positions"


@pytest.mark.parametrize("result", [None, False, 0])
def test_initialization_return_values_do_not_define_success(rig: Rig, monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    strategy = rig.add()

    def initialize() -> object:
        assert strategy.trading is False
        assert strategy.buy(SYMBOL, 4, 100) == []
        return result

    monkeypatch.setattr(strategy, "on_init", initialize)
    rig.initialize(strategy)
    assert strategy.inited is True and strategy.trading is False
    assert len(rig.gateway.subscriptions) == 1
    assert rig.gateway.requests == []


def test_nested_initialization_error_cannot_report_success(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()

    def failure() -> None:
        raise RuntimeError("nested initialization failure")

    def initialize() -> None:
        assert rig.engine.call_strategy_func(strategy, failure) is None

    monkeypatch.setattr(strategy, "on_init", initialize)
    rig.initialize(strategy)
    assert strategy.inited is False and strategy.trading is False
    assert rig.gateway.subscriptions == []
    assert rig.states(strategy)[-1] == [False, False]


def test_ui_recovers_after_startup_failure(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    widget = rig.ui.managers[strategy.strategy_name]
    strategy.start_exception = RuntimeError("startup fails once")
    widget.start_button.click()
    rig.flush()
    assert rig.buttons(strategy) == [True, False, False, True, True]
    strategy.start_exception = None
    widget.init_button.click()
    rig.engine.init_executor.submit(lambda: None).result(timeout=4)
    rig.flush()
    assert rig.buttons(strategy) == [False, True, False, True, True]
    widget.start_button.click()
    rig.flush()
    assert strategy.trading is True and strategy.start_calls == 2


def test_runtime_failure_updates_ui_and_allows_pending_cleanup(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 200)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.gateway.partial_fill(order.orderid, "before-error", 100)
    rig.flush()
    strategy.tick_exception = RuntimeError("runtime callback fails")
    rig.tick()
    assert rig.states(strategy)[-1] == [False, False], "Callback errors must emit a state event"
    assert rig.buttons(strategy) == [True, False, True, True, False]
    assert rig.engine.remove_strategy(strategy.strategy_name) is False
    widget = rig.ui.managers[strategy.strategy_name]
    rig.gateway.defer_cancellation = True
    widget.stop_button.click()
    rig.flush()
    assert strategy.active_orderids == {identity}
    assert rig.buttons(strategy) == [True, False, True, True, False]
    assert strategy.stop_calls == 0
    assert rig.engine.orderid_strategy_map[identity] is strategy
    persisted = json.loads((TEMP_DIR / rig.engine.data_filename).read_text(encoding="utf-8"))
    assert persisted[strategy.strategy_name]["pos_data"][SYMBOL] == 100
    # Deliver the acknowledgement only after the stop event has reached Qt.
    terminal = replace(rig.gateway.orders[order.orderid], status=Status.CANCELLED)
    rig.gateway.orders[order.orderid] = terminal
    rig.gateway.on_order(terminal)
    rig.flush()
    assert not strategy.active_orderids
    assert rig.buttons(strategy) == [True, False, False, True, True]


def test_reinitialization_does_not_overwrite_late_fills_with_old_saved_positions(rig: Rig) -> None:
    strategy = rig.add()
    save_json(rig.engine.data_filename, {
        strategy.strategy_name: {"pos_data": {SYMBOL: 100}, "target_data": {SYMBOL: 300}},
    })
    rig.engine.load_strategy_data()
    rig.initialize(strategy)
    assert strategy.get_pos(SYMBOL) == 100
    rig.engine.start_strategy(strategy.strategy_name)
    identity = strategy.buy(SYMBOL, 4, 300)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.gateway.partial_fill(order.orderid, "first", 100)
    rig.flush()
    assert strategy.get_pos(SYMBOL) == 200
    assert rig.engine.strategy_data[strategy.strategy_name]["pos_data"][SYMBOL] == 100
    strategy.tick_exception = RuntimeError("failure after a newer fill")
    rig.tick()
    rig.initialize(strategy)
    assert strategy.inited is True and strategy.trading is False
    assert strategy.get_pos(SYMBOL) == 200, "An in-process recovery must retain newer trade evidence"
    assert identity in strategy.active_orderids
    assert rig.engine.orderid_strategy_map[identity] is strategy


def test_concurrent_start_requests_call_start_once(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_release.clear()
    first = threading.Thread(target=rig.engine.start_strategy, args=(strategy.strategy_name,))
    first.start()
    try:
        assert strategy.start_entered.wait(4)
        second = threading.Thread(target=rig.engine.start_strategy, args=(strategy.strategy_name,))
        second.start()
        second.join(1)
        assert not second.is_alive(), "Duplicate commands must return without waiting on user code"
        assert strategy.start_calls == 1
    finally:
        strategy.start_release.set()
        first.join(4)
        if "second" in locals():
            second.join(4)
    assert not first.is_alive()
    rig.flush()
    assert strategy.inited is True and strategy.trading is True


def test_stop_during_start_prevents_late_activation(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_release.clear()
    worker = threading.Thread(target=rig.engine.start_strategy, args=(strategy.strategy_name,))
    worker.start()
    try:
        assert strategy.start_entered.wait(4)
        rig.engine.stop_strategy(strategy.strategy_name)
        assert strategy.trading is False
    finally:
        strategy.start_release.set()
        worker.join(4)
    assert not worker.is_alive()
    rig.flush()
    assert strategy.inited is True and strategy.trading is False
    assert strategy.stop_calls == 0
    assert rig.states(strategy)[-1] == [True, False]
    rig.engine.start_strategy(strategy.strategy_name)
    assert strategy.start_calls == 2 and strategy.trading is True


def test_reentrant_start_and_stop_do_not_overlap_callbacks(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    calls: list[str] = []

    def start() -> None:
        calls.append("start")
        assert len(calls) == 1, "Reentrant startup must not invoke the same callback again"
        rig.engine.start_strategy(strategy.strategy_name)
        rig.engine.stop_strategy(strategy.strategy_name)

    monkeypatch.setattr(strategy, "on_start", start)
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert calls == ["start"]
    assert strategy.trading is False and strategy.inited is True
    assert strategy.stop_calls == 0


def test_busy_strategy_does_not_block_another_strategy(rig: Rig) -> None:
    busy = rig.add("busy")
    healthy = rig.add("healthy")
    rig.initialize(busy)
    rig.initialize(healthy)
    busy.start_release.clear()
    first = threading.Thread(target=rig.engine.start_strategy, args=(busy.strategy_name,))
    first.start()
    second = threading.Thread(target=rig.engine.start_strategy, args=(healthy.strategy_name,))
    try:
        assert busy.start_entered.wait(4)
        second.start()
        second.join(1)
        assert not second.is_alive(), "A global lifecycle lock must not cover strategy callbacks"
        assert healthy.trading is True
    finally:
        busy.start_release.set()
        first.join(4)
        if second.ident is not None:
            second.join(4)
    assert not first.is_alive()


def test_duplicate_init_and_removal_while_initializing_are_rejected(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def initialize() -> None:
        calls.append("init")
        entered.set()
        assert release.wait(4)

    monkeypatch.setattr(strategy, "on_init", initialize)
    rig.engine.init_strategy(strategy.strategy_name)
    try:
        assert entered.wait(4)
        rig.engine.init_strategy(strategy.strategy_name)
        assert rig.engine.remove_strategy(strategy.strategy_name) is False
    finally:
        release.set()
        rig.engine.init_executor.submit(lambda: None).result(timeout=4)
    rig.flush()
    assert calls == ["init"]
    assert strategy.inited is True and strategy.trading is False


def test_stop_during_init_prevents_false_ready_state(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    entered = threading.Event()
    release = threading.Event()

    def initialize() -> None:
        entered.set()
        assert release.wait(4)

    monkeypatch.setattr(strategy, "on_init", initialize)
    rig.engine.init_strategy(strategy.strategy_name)
    try:
        assert entered.wait(4)
        rig.engine.stop_strategy(strategy.strategy_name)
    finally:
        release.set()
        rig.engine.init_executor.submit(lambda: None).result(timeout=4)
    rig.flush()
    assert strategy.inited is False and strategy.trading is False
    assert rig.gateway.subscriptions == []
    assert strategy.stop_calls == 0
    assert rig.states(strategy)[-1] == [False, False]


def test_failed_reinitialization_cannot_clear_pending_cleanup(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    strategy.tick_exception = RuntimeError("runtime failure")
    rig.tick()
    strategy.init_exception = RuntimeError("reinitialization fails too")
    rig.initialize(strategy)
    assert strategy.inited is False and strategy.trading is False
    before = len(rig.events)
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert len(rig.events) == before + 1


def test_callback_helper_keeps_none_return_and_original_argument_rules(rig: Rig) -> None:
    strategy = rig.add()
    observed: list[object] = []
    assert rig.engine.call_strategy_func(strategy, lambda: observed.append("empty"), 0) is None
    assert rig.engine.call_strategy_func(strategy, observed.append, "nonempty") is None
    assert observed == ["empty", "nonempty"]


def test_subscription_exception_cannot_declare_initialization_success(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()

    def subscribe(req: object) -> None:
        raise RuntimeError("subscription adapter failed")

    monkeypatch.setattr(rig.gateway, "subscribe", subscribe)
    rig.initialize(strategy)
    assert strategy.inited is False and strategy.trading is False
    assert rig.states(strategy)[-1] == [False, False]
    assert any("subscription adapter failed" in line for line in rig.logs)


def test_failed_saved_data_restore_keeps_original_file(rig: Rig) -> None:
    strategy = rig.add()
    # This is deliberately invalid saved input, confined to the disposable fixture.
    rig.engine.strategy_data[strategy.strategy_name] = {"pos_data": 500}
    save_json(rig.engine.data_filename, rig.engine.strategy_data)
    path = TEMP_DIR / rig.engine.data_filename
    before = path.read_bytes()
    rig.initialize(strategy)
    assert strategy.inited is False and strategy.trading is False
    rig.engine.stop_strategy(strategy.strategy_name)
    assert path.read_bytes() == before
    assert rig.states(strategy)[-1] == [False, False]


def test_saved_flags_cannot_enable_trading_during_initialization(rig: Rig) -> None:
    strategy = rig.add()
    rig.engine.strategy_data[strategy.strategy_name] = {
        "inited": True, "trading": True, "pos_data": {SYMBOL: 500}, "target_data": {SYMBOL: 600},
    }
    rig.initialize(strategy)
    assert strategy.inited is True and strategy.trading is False
    assert strategy.get_pos(SYMBOL) == 500 and strategy.get_target(SYMBOL) == 600
    assert strategy.buy(SYMBOL, 4, 100) == []
    assert rig.gateway.requests == []


def test_closing_during_start_prevents_late_activation(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_release.clear()
    worker = threading.Thread(target=rig.engine.start_strategy, args=(strategy.strategy_name,))
    worker.start()
    try:
        assert strategy.start_entered.wait(4)
        rig.engine.close()
    finally:
        strategy.start_release.set()
        worker.join(4)
    assert not worker.is_alive()
    assert strategy.trading is False
    calls = strategy.start_calls
    rig.engine.start_strategy(strategy.strategy_name)
    assert strategy.start_calls == calls


def test_stop_callback_reentry_calls_stop_once(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    rig.engine.start_strategy(strategy.strategy_name)
    calls: list[str] = []

    def stop() -> None:
        calls.append("stop")
        assert len(calls) == 1
        assert strategy.trading is True, "Keep the native on_stop callback's permission ordering"
        rig.engine.stop_strategy(strategy.strategy_name)
        rig.engine.start_strategy(strategy.strategy_name)

    monkeypatch.setattr(strategy, "on_stop", stop)
    rig.engine.stop_strategy(strategy.strategy_name)
    assert calls == ["stop"]
    assert strategy.trading is False and strategy.inited is True


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
def test_base_exception_propagates_and_releases_operation(rig: Rig, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]) -> None:
    strategy = rig.add()
    rig.initialize(strategy)

    def terminate() -> None:
        raise error("intentional callback interruption")

    monkeypatch.setattr(strategy, "on_start", terminate)
    with pytest.raises(error, match="intentional callback interruption"):
        rig.engine.start_strategy(strategy.strategy_name)
    monkeypatch.setattr(strategy, "on_start", lambda: None)
    rig.engine.start_strategy(strategy.strategy_name)
    assert strategy.trading is True and strategy.inited is True


def test_failed_stop_emits_cleanup_event_without_new_orders(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    strategy.start_exception = RuntimeError("startup failure")
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    before = len(rig.events)
    rig.engine.stop_strategy(strategy.strategy_name)
    rig.flush()
    assert len(rig.events) == before + 1
    assert rig.states(strategy)[-1] == [False, False]
    assert strategy.stop_calls == 0
    assert rig.gateway.requests == []


@pytest.mark.parametrize("running", [False, True])
def test_new_ui_widget_uses_current_strategy_state(rig: Rig, running: bool) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    if running:
        rig.engine.start_strategy(strategy.strategy_name)
    data = strategy.get_data()
    widget = StrategyManager(rig.ui, rig.engine, data)
    try:
        observed = [button.isEnabled() for button in [
            widget.init_button, widget.start_button, widget.stop_button, widget.edit_button, widget.remove_button,
        ]]
        assert observed == ([False, False, True, False, False] if running else [False, True, False, True, True])
    finally:
        widget.deleteLater()
        rig.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_start_callback_can_revoke_initialization_without_enabling_trading(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = rig.add()
    rig.initialize(strategy)

    def start() -> None:
        strategy.inited = False

    monkeypatch.setattr(strategy, "on_start", start)
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    assert strategy.inited is False and strategy.trading is False
    assert rig.states(strategy)[-1] == [False, False]


@pytest.mark.parametrize("method", ["start_strategy", "stop_strategy", "remove_strategy", "_init_strategy"])
def test_unknown_strategy_keeps_native_key_error(rig: Rig, method: str) -> None:
    with pytest.raises(KeyError, match="missing_strategy"):
        getattr(rig.engine, method)("missing_strategy")


def test_custom_strategy_does_not_need_to_be_hashable(rig: Rig) -> None:
    class UnhashableStrategy(EngineeringStrategy):
        def __eq__(self, other: object) -> bool:
            return self is other

    rig.engine.classes["EngineeringStrategy"] = UnhashableStrategy
    strategy = rig.add()
    try:
        rig.initialize(strategy)
        assert strategy.inited is True and strategy.trading is False
        rig.engine.start_strategy(strategy.strategy_name)
        assert strategy.trading is True
        rig.engine.stop_strategy(strategy.strategy_name)
        assert strategy.inited is True and strategy.trading is False
        assert rig.engine.remove_strategy(strategy.strategy_name) is True
    finally:
        # A broken candidate may fail before removal; still release the disposable fixture.
        rig.engine.strategies.pop(strategy.strategy_name, None)


def test_error_from_removed_instance_does_not_change_reused_name(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    old = rig.add()
    rig.initialize(old)
    rig.engine.start_strategy(old.strategy_name)
    identity = old.buy(SYMBOL, 4, 100)[0]
    order = next(iter(rig.gateway.orders.values()))
    rig.gateway.on_order(order)
    rig.flush()
    rig.engine.stop_strategy(old.strategy_name)
    rig.flush()
    assert not old.active_orderids
    assert rig.engine.remove_strategy(old.strategy_name) is True
    assert rig.engine.orderid_strategy_map[identity] is old
    fresh = rig.add(old.strategy_name)
    rig.initialize(fresh)
    rig.engine.start_strategy(fresh.strategy_name)
    rig.flush()

    def fail_late_order(order: object) -> None:
        raise RuntimeError("late callback belongs to the removed instance")

    monkeypatch.setattr(old, "update_order", fail_late_order)
    rig.gateway.on_order(rig.gateway.orders[order.orderid])
    rig.flush()
    assert fresh.inited is True and fresh.trading is True
    assert rig.buttons(fresh) == [False, False, True, False, False]


def test_buffered_event_from_removed_instance_is_ignored(rig: Rig) -> None:
    old = rig.add()
    rig.initialize(old)
    saved_event = deepcopy(rig.events[-1])
    assert rig.engine.remove_strategy(old.strategy_name) is True
    fresh = rig.add(old.strategy_name)
    rig.initialize(fresh)
    rig.engine.start_strategy(fresh.strategy_name)
    rig.flush()
    # Re-delivery models an event already buffered before the old instance was removed.
    rig.main.event_engine.put(Event(EVENT_PORTFOLIO_STRATEGY, saved_event))
    rig.flush()
    assert fresh.inited is True and fresh.trading is True
    assert rig.buttons(fresh) == [False, False, True, False, False]


def test_buffered_older_state_from_current_instance_is_ignored(rig: Rig) -> None:
    strategy = rig.add()
    rig.initialize(strategy)
    saved_event = deepcopy(rig.events[-1])
    rig.engine.start_strategy(strategy.strategy_name)
    rig.flush()
    rig.main.event_engine.put(Event(EVENT_PORTFOLIO_STRATEGY, saved_event))
    rig.flush()
    assert strategy.inited is True and strategy.trading is True
    assert rig.buttons(strategy) == [False, False, True, False, False]
