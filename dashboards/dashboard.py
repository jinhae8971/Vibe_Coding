import sqlite3
import streamlit as st


def load_data(db_path: str = "trades.sqlite3"):
    conn = sqlite3.connect(db_path)
    trades = conn.execute("SELECT * FROM trades").fetchall()
    positions = conn.execute("SELECT * FROM positions").fetchall()
    conn.close()
    return trades, positions


def main() -> None:
    st.title("Trading Bot Dashboard")
    trades, positions = load_data()

    st.subheader("Trades")
    st.table(trades)

    st.subheader("Positions")
    st.table(positions)


if __name__ == "__main__":
    main()
