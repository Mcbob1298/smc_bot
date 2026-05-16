# Archived Code — BTC/CCXT Support

**Date d'archivage :** 2026-05-16

## Raison

V1 du bot se concentre exclusivement sur XAUUSD via MetaTrader 5.
Le support BTC/crypto via CCXT est fonctionnel mais non nécessaire pour V1.
Décision stratégique : focus sur un marché où SMC est historiquement plus efficace,
simplification architecturale, optimisation focalisée du backtest.

## État du code au moment de l'archivage

- **Tous les tests passaient** : 19/19 (6 connection + 13 download)
- Code fonctionnel avec :
  - Retry automatique (tenacity, 3 attempts, exponential backoff)
  - Symbol mapping (BTCUSDT → BTC/USDT)
  - Pagination forward (1000 bars/request)
  - Gap detection (crypto 24/7, gaps = maintenance)
  - Incomplete bar filtering
  - Partial data warning
  - disconnect() gère l'absence de close()

## Contenu

```
_archived/
├── README.md                 # ce fichier
├── data/ingestion/
│   └── ccxt_loader.py        # CCXTLoader class
├── scripts/
│   └── test_ccxt_loader.py   # Demo script (live Binance)
└── tests/
    └── test_ccxt_loader.py   # 19 unit tests (mocked)
```

## Procédure de réactivation (V2)

1. Déplacer les fichiers à leur emplacement original :
   ```bash
   mv _archived/data/ingestion/ccxt_loader.py data/ingestion/
   mv _archived/tests/test_ccxt_loader.py tests/
   mv _archived/scripts/test_ccxt_loader.py scripts/
   ```

2. Dans `pyproject.toml` :
   - Déplacer `ccxt>=4.0` de `[project.optional-dependencies].crypto` vers `dependencies`
   - Retirer `_archived` de `norecursedirs`, `exclude` (ruff/mypy)

3. Dans `config/settings.py` :
   - Réactiver `binance_api_key`, `binance_api_secret`, `ccxt_symbol_map`

4. Installer et tester :
   ```bash
   uv sync
   uv run pytest tests/test_ccxt_loader.py -v
   ```

5. Ajouter BTCUSDT à `symbols` dans settings et adapter strategy.py
