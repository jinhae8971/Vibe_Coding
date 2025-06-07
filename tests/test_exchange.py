from exchange import BinanceExchange


def test_binance_exchange_interface():
    ex = BinanceExchange("key", "secret")
    assert hasattr(ex, "get_balance")
    assert hasattr(ex, "fetch_position")
    assert hasattr(ex, "place_order")
