# Open Questions

Methodological ambiguities and design decisions to resolve before implementation.

---

### [Q-001] Définition "liquidité prise avant l'OB"
- **Context**: Un OB valide selon Kasper doit "idéalement" avoir pris de la liquidité avant. Mais combien de bougies en arrière chercher ? Et quel type de liquidité (equal highs/lows uniquement, ou trendline aussi, ou tout pool identifié) ?
- **Default choice**: Chercher dans les 20 dernières bougies (même TF que l'OB) un sweep d'un swing antérieur identifié (equal H/L ou single swing). Trendline liquidity aussi acceptée si détectée.
- **Alternatives**: (A) Fenêtre fixe de N bougies configurable. (B) Pas de limite de lookback, chercher le pool de liquidité le plus proche dans le passé. (C) Rendre ce critère optionnel (bonus scoring plutôt que filtre binaire).
- **Status**: Open

---

### [Q-002] Définition "première touche" d'un OB (mitigation)
- **Context**: On ne trade un OB qu'au "premier retest". Mais qu'est-ce qui constitue un retest ? Une mèche qui pique dans la zone sans clôturer dedans compte-t-elle ? Ou faut-il une clôture ?
- **Default choice**: Le premier retest = première bougie dont le high (pour un bearish OB) ou le low (pour un bullish OB) ENTRE dans la zone OB. Même une mèche seule déclenche la possibilité d'entrée (on attend ensuite la confirmation LTF). L'OB est considéré "touché" et ne sera plus retradé.
- **Alternatives**: (A) Seule une clôture dans l'OB constitue le retest. (B) Entrée directe dès la mèche sans attendre confirmation LTF.
- **Status**: Open

---

### [Q-003] Zone OB : mèche complète ou corps uniquement ?
- **Context**: La bougie qui forme l'OB définit une zone. Prend-on le range complet (high-low) ou seulement le corps (open-close) ?
- **Default choice**: `full_range` (high-low). Le corps seul serait trop restrictif et raterait des entrées. La zone mèche est justifiée car les mèches représentent de l'intérêt institutionnel.
- **Alternatives**: (A) Corps uniquement (plus conservateur, SL plus serré). (B) Corps + 50% de la mèche comme compromis. (C) Configurable par le paramètre `body_or_full_range`.
- **Status**: Open

---

### [Q-004] ChoCh LTF — quel niveau de structure est requis ?
- **Context**: Sur LTF (M1/M5), on attend un ChoCh dans le sens du biais HTF. Mais la structure LTF est très bruitée. Le ChoCh doit-il casser le dernier micro-swing immédiat, ou un swing "significatif" (filtré ATR) ?
- **Default choice**: ChoCh LTF doit casser le dernier swing SIGNIFICATIF (filtré ATR avec `atr_filter_ratio`) en clôture. Un micro-swing non-significatif cassé ne compte pas.
- **Alternatives**: (A) Dernier micro-swing brut (sans filtre ATR) — plus de signaux mais plus de bruit. (B) Exiger un BOS après le ChoCh pour confirmer (plus conservateur, moins de trades).
- **Status**: Open

---

### [Q-005] Biais HTF — H4 seul ou alignement H4 + Daily requis ?
- **Context**: Le biais est déterminé en HTF. Kasper utilise H4 principalement, mais mentionne Daily "en confirmation". Le Daily est-il obligatoire ou optionnel ?
- **Default choice**: Biais primaire = dernière structure H4 (BOS = continuation, ChoCh = renversement). Daily en filtre optionnel : si Daily et H4 sont contradictoires, ne pas trader (mode "no bias"). Si alignés, signal plus fort.
- **Alternatives**: (A) H4 seul, ignorer Daily. (B) Daily obligatoire — ne trader que si H4 + Daily alignés. (C) Daily uniquement pour le biais directionnel, H4 uniquement pour timing.
- **Status**: Open

---

### [Q-006] Multi-position dans la même killzone ?
- **Context**: Si un premier trade est stoppé dans une killzone, peut-on reprendre un deuxième trade dans la même session ? Kasper mentionne "un setup par killzone" dans certaines vidéos mais pas systématiquement.
- **Default choice**: Maximum 1 trade par killzone par session. Si stoppé, attendre la prochaine KZ. Réduit l'overtrading et la revenge trading.
- **Alternatives**: (A) Maximum 2 trades par KZ (permet un retry). (B) Pas de limite mais respecter `max_concurrent_trades`. (C) Configurable via paramètre.
- **Status**: Open

---

### [Q-007] Gestion du weekend gap (XAU ferme vendredi, BTC 24/7)
- **Context**: XAUUSD ferme le vendredi ~23h et rouvre dimanche ~23h (heure Paris). Un gap significatif peut se former. Doit-on clôturer les positions le vendredi avant fermeture ? BTC n'a pas ce problème.
- **Default choice**: Pour XAU, clôturer toute position ouverte 30 min avant la fermeture du marché vendredi (22h30 Paris). Pour BTC, pas de contrainte weekend mais réduire la taille si un trade est ouvert pendant le weekend (liquidité réduite).
- **Alternatives**: (A) Laisser les positions ouvertes (accepter le gap risk). (B) Ne pas ouvrir de nouveau trade après 18h le vendredi. (C) Configurable par paramètre.
- **Status**: Open

---

### [Q-008] Source du calendrier économique
- **Context**: ForexFactory est la source classique mais n'a pas d'API officielle (scraping fragile). Investing.com est une alternative. MQL5 a un calendrier intégré aussi.
- **Default choice**: Utiliser le calendrier économique intégré de MQL5/MT5 (via `MetaTrader5.calendar_get()`) pour XAU en priorité. Backup = scraping ForexFactory hebdomadaire avec cache local. Pour BTC, cross-check avec les events crypto (pas sur ForexFactory).
- **Alternatives**: (A) ForexFactory uniquement (XML/scraping). (B) Investing.com API non-officielle. (C) Service payant (FXStreet API, TradingEconomics). (D) Fichier CSV maintenu manuellement.
- **Status**: Open

---

### [Q-009] Tolérance "equal highs/lows" — unité de mesure
- **Context**: Les equal highs/lows sont détectés quand plusieurs swings sont au "même niveau". Quelle tolérance exacte ? En ATR (dynamique) ou en valeur absolue ?
- **Default choice**: Tolérance = `ATR(14) × 0.1` (dynamique, s'adapte à la volatilité). Deux swings sont "equal" si `|price_A - price_B| <= tolerance`. Calculé sur le timeframe du détecteur.
- **Alternatives**: (A) Valeur absolue fixe par instrument (XAU: 0.50$, BTC: 50$). (B) Pourcentage du prix (0.02%). (C) Nombre de ticks broker.
- **Status**: Open

---

### [Q-010] OB contenant un FVG interne — zone d'entrée
- **Context**: Un OB peut contenir un FVG dans son range (l'impulsion qui suit l'OB crée un FVG qui chevauche partiellement l'OB). L'entrée optimale est-elle sur la zone OB complète ou spécifiquement sur le FVG interne ?
- **Default choice**: Entrée sur la zone OB complète. Le FVG interne est un bonus de confluence qui augmente la probabilité mais ne modifie pas la zone d'entrée. Le SL est calculé sur l'OB, pas le FVG.
- **Alternatives**: (A) Entrée uniquement sur le FVG interne (SL plus serré, meilleur RR mais plus de trades ratés). (B) Entrée sur l'overlap OB∩FVG. (C) Si FVG interne existe, l'utiliser comme zone d'entrée avec SL élargi à l'OB.
- **Status**: Open

---

### [Q-011] Invalidation OB — clôture complète au-delà ou mèche suffisante ?
- **Context**: Les specs disent "si clôture de bougie au-delà de l'OB → invalidé". Mais "au-delà" = au-delà du bord extrême de l'OB (mèche comprise) ? Ou au-delà du corps de l'OB ?
- **Default choice**: Invalidation = clôture de bougie au-delà du bord extrême (full range) de l'OB. Pour un bullish OB, invalidé si close < OB_low. Pour un bearish OB, invalidé si close > OB_high.
- **Alternatives**: (A) Invalider dès qu'une mèche dépasse (plus strict). (B) Tolérance de X% de dépassement avant invalidation. (C) Invalider uniquement si 2 clôtures consécutives au-delà.
- **Status**: Open

---

### [Q-012] Combien de temps un FVG reste-t-il valide ?
- **Context**: Un FVG non-comblé (unfilled) reste-il valide indéfiniment ? Ou a-t-il une durée de vie maximale comme les OB ?
- **Default choice**: Un FVG reste valide tant qu'il n'est pas comblé (fill = une bougie clôture dans le gap et le traverse complètement). Pas de limite temporelle. Cependant, seuls les FVG des 50 dernières bougies sont considérés pour l'association avec un OB.
- **Alternatives**: (A) Durée de vie max configurable (ex: 100 bougies). (B) FVG partiellement comblé (50%+) est invalidé. (C) Validité liée au timeframe (FVG H4 valide plus longtemps que FVG M15).
- **Status**: Open

---

### [Q-013] Breaker Block — implémentation ou hors scope initial ?
- **Context**: Les specs mentionnent qu'un OB invalidé "devient Breaker Block potentiel". Faut-il implémenter les Breaker Blocks dès le départ ou les ajouter plus tard ?
- **Default choice**: Hors scope pour l'étape 3 (détecteurs). Marquer les OB invalidés avec un flag `is_broken=True` pour pouvoir les transformer en Breaker Blocks dans une version ultérieure.
- **Alternatives**: (A) Implémenter immédiatement (complexifie le code). (B) Ignorer complètement les Breaker Blocks.
- **Status**: Open

---

### [Q-014] Spread dynamique vs fixe pour calcul d'entrée
- **Context**: Les specs disent "tenir compte du spread broker dans le calcul d'entrée". Le spread XAU varie (2-5 pips selon horaire). Utilise-t-on le spread live ou un spread fixe moyen pour le backtest ?
- **Default choice**: En backtest, utiliser un spread fixe conservateur par session (London: 2.5 pips, NY: 2.0 pips, hors-session: 4.0 pips pour XAU). En live, utiliser le spread bid/ask réel.
- **Alternatives**: (A) Spread fixe unique (3 pips XAU, 0.03% BTC). (B) Spread dynamique historique si données dispo. (C) Ajouter un buffer de slippage en plus du spread.
- **Status**: Open
