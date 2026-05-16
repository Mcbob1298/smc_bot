# Open Questions

Methodological ambiguities and design decisions resolved before implementation.

---

### [Q-001] Définition "liquidité prise avant l'OB"
- **Context**: Un OB valide selon Kasper doit "idéalement" avoir pris de la liquidité avant. Mais combien de bougies en arrière chercher ? Et quel type de liquidité (equal highs/lows uniquement, ou trendline aussi, ou tout pool identifié) ?
- **Decision**: Lookback 20 bougies (même TF que l'OB). Accepter tout type de liquidité : swing antérieur, equal H/L, ou trendline liquidity. Paramètre `require_prior_liquidity_sweep` activable/désactivable pour A/B test.
- **Rationale**: 20 bougies couvre environ 1 session de trading sur M15 — au-delà, la relation causale avec l'OB devient trop faible.
- **Status**: Resolved

---

### [Q-002] Définition "première touche" d'un OB (mitigation)
- **Context**: On ne trade un OB qu'au "premier retest". Mais qu'est-ce qui constitue un retest ? Une mèche qui pique dans la zone sans clôturer dedans compte-t-elle ? Ou faut-il une clôture ?
- **Decision**: Mèche qui entre dans l'OB = déclenche le check d'entrée (on passe en LTF pour confirmation). Clôture dans l'OB = mitigation effective (OB consommé, ne sera plus retradé). `wick_touch_counts_as_mitigation=False` confirmé.
- **Rationale**: La mèche montre l'intérêt institutionnel mais ne constitue pas une absorption complète. La clôture prouve que les ordres ont été remplis.
- **Status**: Resolved

---

### [Q-003] Zone OB : mèche complète ou corps uniquement ?
- **Context**: La bougie qui forme l'OB définit une zone. Prend-on le range complet (high-low) ou seulement le corps (open-close) ?
- **Decision**: Default `full_range` (high-low). Paramètre configurable avec 3 valeurs possibles : `"full_range"`, `"body_only"`, `"body_plus_half_wick"` pour A/B testing en backtest.
- **Rationale**: Full range est plus conservateur (zone plus large, SL plus large mais moins de trades ratés). On pourra comparer les 3 modes en backtest.
- **Status**: Resolved

---

### [Q-004] ChoCh LTF — quel niveau de structure est requis ?
- **Context**: Sur LTF (M1/M5), on attend un ChoCh dans le sens du biais HTF. Mais la structure LTF est très bruitée. Le ChoCh doit-il casser le dernier micro-swing immédiat, ou un swing "significatif" (filtré ATR) ?
- **Decision**: ChoCh LTF avec filtre ATR activé par défaut mais ratio plus bas (0.15) que HTF/MTF (0.3). Cela filtre le bruit M1 tout en gardant de la sensibilité.
- **Rationale**: Ratio 0.15 permet de filtrer les micro-oscillations de 1-2 pips sur M1 XAU qui ne représentent pas de vrais mouvements structurels, tout en restant réactif.
- **Status**: Resolved

---

### [Q-005] Biais HTF — H4 seul ou alignement H4 + Daily requis ?
- **Context**: Le biais est déterminé en HTF. Kasper utilise H4 principalement, mais mentionne Daily "en confirmation". Le Daily est-il obligatoire ou optionnel ?
- **Decision**: H4 = biais primaire. Daily = filtre optionnel (`daily_alignment_mode="optional_filter"`). Si Daily contredit H4, pas de trade. Si Daily neutre ou aligné, trade autorisé.
- **Rationale**: Évite de trader contre la tendance de fond tout en ne paralysant pas le système quand le Daily est en consolidation.
- **Status**: Resolved

---

### [Q-006] Multi-position dans la même killzone ?
- **Context**: Si un premier trade est stoppé dans une killzone, peut-on reprendre un deuxième trade dans la même session ?
- **Decision**: Maximum 1 trade par killzone. Si stoppé, attendre la prochaine KZ.
- **Rationale**: Anti-revenge trading. Un stop dans une KZ signifie que notre lecture de la session était incorrecte — persister augmente le drawdown sans edge supplémentaire.
- **Status**: Resolved — Updated (V1 scope: XAU only)

---

### [Q-007] Gestion du weekend gap (XAU ferme vendredi)
- **Context**: XAUUSD ferme le vendredi ~23h et rouvre dimanche ~23h (heure Paris). Un gap significatif peut se former.
- **Decision**: Clôturer toute position à 22h30 Paris vendredi. Pas de nouveau trade après 18h00 Paris vendredi.
- **Rationale**: Le gap risk du weekend est non-rémunéré et peut effacer plusieurs jours de gains. Mieux vaut être flat.
- **Status**: Resolved — Updated (V1 scope: XAU only)

---

### [Q-008] Source du calendrier économique
- **Context**: ForexFactory est la source classique mais n'a pas d'API officielle. MQL5 a un calendrier intégré.
- **Decision**: MT5 `calendar_get()` en source primaire. ForexFactory scraping hebdomadaire en fallback automatique si MT5 indisponible. Architecture `EconCalendar` avec backend pluggable.
- **Rationale**: MT5 calendar est fiable, structuré, et ne nécessite pas de scraping fragile. ForexFactory reste utile comme backup et pour validation croisée.
- **Status**: Resolved

---

### [Q-009] Tolérance "equal highs/lows" — unité de mesure
- **Context**: Les equal highs/lows sont détectés quand plusieurs swings sont au "même niveau". Quelle tolérance exacte ?
- **Decision**: Tolérance = `ATR(14) × 0.1` (dynamique). Calculé sur le timeframe du détecteur.
- **Rationale**: Une tolérance dynamique s'adapte à la volatilité de l'instrument et du régime de marché, contrairement à une valeur fixe qui serait inadaptée quand la volatilité change.
- **Status**: Resolved

---

### [Q-010] OB contenant un FVG interne — zone d'entrée
- **Context**: Un OB peut contenir un FVG dans son range. L'entrée optimale est-elle sur la zone OB complète ou spécifiquement sur le FVG interne ?
- **Decision**: Entrée sur zone OB complète. FVG interne = bonus de confluence stocké comme flag `has_internal_fvg` sur l'OB. Non utilisé pour l'entrée en V1, conservé pour scoring/analyse V2.
- **Rationale**: Simplifier la V1. Le FVG interne sera exploité dans le scoring de qualité des setups en V2 pour prioriser les meilleurs OB.
- **Status**: Resolved

---

### [Q-011] Invalidation OB — clôture complète au-delà ou mèche suffisante ?
- **Context**: Les specs disent "si clôture de bougie au-delà de l'OB → invalidé". Mais "au-delà" = au-delà du bord extrême ?
- **Decision**: Invalidation = clôture de bougie au-delà du bord extrême (full range). Bullish OB invalidé si `close < OB_low`. Bearish OB invalidé si `close > OB_high`.
- **Rationale**: La clôture prouve que le niveau a été absorbé et que l'intérêt institutionnel n'a pas tenu. Une mèche seule est du bruit.
- **Status**: Resolved

---

### [Q-012] Combien de temps un FVG reste-t-il valide ?
- **Context**: Un FVG non-comblé reste-il valide indéfiniment ? Ou a-t-il une durée de vie maximale ?
- **Decision**: FVG sans expiration globale. Association à un OB limitée à une fenêtre de 50 bougies (`fvg_association_window_bars=50`). Un FVG est invalidé uniquement quand entièrement comblé (bougie traverse le gap en clôture).
- **Rationale**: Un FVG non-comblé reste un déséquilibre réel du marché. Mais l'association OB↔FVG n'a de sens que dans une fenêtre temporelle restreinte (causalité).
- **Status**: Resolved

---

### [Q-013] Breaker Block — implémentation ou hors scope initial ?
- **Context**: Un OB invalidé "devient Breaker Block potentiel". Faut-il l'implémenter dès le départ ?
- **Decision**: Hors scope V1. Stocker `is_broken: bool` et `broken_at_bar: int | None` sur chaque OB invalidé. Logique Breaker Block prévue V2.
- **Rationale**: Complexité significative (inversion de polarité, nouveaux critères de validité) pour un edge non encore prouvé. On valide d'abord l'OB standard.
- **Status**: Resolved

---

### [Q-014] Spread dynamique vs fixe + slippage pour backtest
- **Context**: Les specs disent "tenir compte du spread broker dans le calcul d'entrée". Le spread XAU varie selon horaire. Le slippage est souvent ignoré mais critique pour les stratégies SMC qui entrent sur des mouvements rapides.
- **Decision**: Spread variable par session + slippage séparé obligatoire.
  - Spread XAU : London KZ = 2.5 pips, NY KZ = 2.0 pips, hors KZ = 4.0 pips
  - Slippage : `0.5 × ATR(M1)` en killzone, `1.0 × ATR(M1)` hors killzone, appliqué à entry ET exit
  - Tous ces paramètres dans `CostsConfig` (configurable pour A/B test)
- **Rationale**: Le slippage est souvent plus impactant que le spread sur les stratégies SMC qui entrent au moment où le marché bouge fort (cassures de structure, sweeps). L'oublier surestime massivement l'edge. Valeurs conservatrices par défaut — à affiner avec données réelles live.
- **Status**: Resolved — Updated (V1 scope: XAU only, BTC costs removed)
