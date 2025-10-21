from decimal import Decimal
from uuid import UUID
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import talib as ta
from typing import Any, Optional
from datetime import datetime, timedelta, timezone
from app.infrastructure import alpaca_service
from sklearn.model_selection import ParameterGrid
from app.domain.position_manager import PositionManager


from app.domain import *

async def plot_strategy(start: datetime, end: datetime):
    
    symbol = "SPY"
    data = pd.read_csv("breakout_test.csv")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Escape-Taste Handler
    def on_key(event):
        if event.key == 'escape':
            plt.close(fig)
            print("Chart geschlossen")
    
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Zeitachse formatieren
    data['datetime'] = pd.to_datetime(data['T'].str.replace("Z", "+00:00"))
    filtered_data = data[(data['datetime'] >= start) & (data['datetime'] <= end)]

    # Verwende gefilterte Daten falls vorhanden, sonst alle Daten
    if len(filtered_data) > 0:
        plot_data = filtered_data.copy()
        print(f"Zeige {len(plot_data)} Datenpunkte im Zeitbereich {start} bis {end}")
    else:
        plot_data = data.copy()
        print(f"Warnung: Keine Daten im Zeitbereich {start} bis {end} gefunden. Zeige alle {len(plot_data)} verfügbaren Daten.")
    
    # Reset Index für kontinuierliche x-Achse ohne Lücken
    plot_data = plot_data.reset_index(drop=True)
    
    # Verwende numerische Indizes für x-Achse (keine Lücken)
    x_values = range(len(plot_data))
    
    # Plotte mit numerischen Indizes
    ax.plot(x_values, plot_data['C'], label='Schlusskurs', alpha=0.7, color='blue', linewidth=1.5)
    ax.plot(x_values, plot_data['donchian_low'], label='Donchian Low', linestyle='--', color='purple', linewidth=2)
    ax.plot(x_values, plot_data['donchian_high'], label='Donchian High', linestyle='--', color='orange', linewidth=2)
    
    # Kanal-Bereich füllen
    ax.fill_between(x_values, plot_data['donchian_high'], plot_data['donchian_low'], 
                   alpha=0.1, color='gray', label='Donchian Channel')

    # Signale plotten
    buy_signals = plot_data[plot_data['signal'] == SignalEnum.BUY.value]
    sell_signals = plot_data[plot_data['signal'] == SignalEnum.SELL.value]
    
    if len(buy_signals) > 0:
        ax.scatter(buy_signals.index, buy_signals['C'], 
                  label='Buy Signal', marker='^', color='green', s=100, zorder=5)
    
    if len(sell_signals) > 0:
        ax.scatter(sell_signals.index, sell_signals['C'], 
                  label='Sell Signal', marker='v', color='red', s=100, zorder=5)

    # Custom x-Achsen Labels (zeige nur bestimmte Zeitpunkte)
    # Berechne Tick-Positionen (z.B. jeden 10. Datenpunkt)
    tick_interval = max(1, len(plot_data) // 10)  # Maximal 10 Labels
    tick_positions = range(0, len(plot_data), tick_interval)
    tick_labels = [plot_data.iloc[i]['datetime'].strftime('%m-%d %H:%M') for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    # Achsen-Grenzen setzen
    ax.set_xlim(0, len(plot_data) - 1)
    ax.set_ylim(plot_data['donchian_low'].min() * 0.999, plot_data['donchian_high'].max() * 1.001)

    # Titel und Legende
    ax.set_title(f'Donchian Breakout Strategie für {symbol}')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Preis ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return

