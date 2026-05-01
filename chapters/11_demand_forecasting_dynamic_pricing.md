# Demand Forecasting & Dynamic Pricing — Uber Airport Rides

## Problem Formulation

Two coupled models: (1) **Demand forecasting**: predict ride requests $D(location, time, horizon)$ for next 5–60 minutes at H3 hexagonal grid resolution. (2) **Dynamic pricing (surge)**: set price multiplier $m$ to balance supply/demand: $m^* = \arg\min |S(m) - D|$ where $S(m)$ = supply attracted at multiplier $m$.

Joint objective: Maximize platform GMV while maintaining rider satisfaction (ETA < threshold) and driver utilization (>70%).

$$\text{Revenue} = \sum_{hex, t} \min(D_{hex,t}, S_{hex,t}(m)) \times \text{base\_fare} \times m_{hex,t}$$

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Spatial resolution | H3 hexagons (resolution 7, ~5.16 km² each) |
| Temporal resolution | 5-minute intervals for short-term, hourly for medium-term |
| Demand signals | Real-time app opens, ride requests, search queries |
| Supply signals | Active drivers, driver heading/ETA, shift patterns |
| External features | Weather (precipitation, temp), events (concerts, sports), flights (airport arrivals) |
| Historical | 2 years of trip data, 1B+ trips |
| Airport-specific | Flight arrival schedules, terminal mapping, pickup queue length |

**Airport-specific features**:
- Flight arrival volume in next 30/60/120 min (from FlightAware API)
- International vs. domestic (international → longer customs → delayed demand)
- Time of day × day of week × season interaction
- Pickup queue length (supply already waiting at airport)
- Surge in neighboring hexagons (demand spillover)

---

## Model Architecture

**Demand Forecasting — Temporal Fusion Transformer (TFT)**:

$$\hat{y}_{t+\tau} = \text{TFT}(y_{t-L:t}, x^{static}, x^{known}_{t:t+\tau}, x^{observed}_{t-L:t})$$

Components:
- Variable selection networks (learn which features matter per time step)
- LSTM encoder-decoder for temporal patterns
- Multi-head attention over encoder states (captures long-range seasonality)
- Quantile output: predicts P10, P50, P90 (uncertainty quantification)

**Alternative: DeepAR** (Amazon, autoregressive):
$$P(y_{t+1:t+T} | y_{1:t}, x) = \prod_{\tau=1}^T P(y_{t+\tau} | y_{1:t+\tau-1}, x)$$

Parameterized as negative binomial distribution (handles count data with overdispersion).

**Dynamic Pricing — Constrained Optimization**:

$$m^* = \arg\max_m \left[ \text{Revenue}(m) - \lambda_1 \cdot \text{WaitTimePenalty}(m) - \lambda_2 \cdot \text{DriverIdlePenalty}(m) \right]$$

Supply elasticity model: $S(m) = S_0 \cdot m^\epsilon$ where $\epsilon$ = supply elasticity (learned from historical surge experiments).

Demand elasticity: $D(m) = D_0 \cdot m^{-\eta}$ where $\eta$ ≈ 0.3–0.5 (riders are somewhat price-sensitive).

Equilibrium: solve $S(m) = D(m)$ → $m^* = (D_0/S_0)^{1/(\epsilon + \eta)}$

---

## Training & Optimization

| Config | Value |
|--------|-------|
| TFT | Adam (lr=1e-3), batch=256 time series, quantile loss |
| Lookback | 2 weeks (capture weekly seasonality) |
| Forecast horizon | 5, 15, 30, 60 min (multi-horizon) |
| Spatial model | One model per city, shared architecture |
| Retraining | Daily (new day's data appended, oldest day dropped) |
| Loss | Quantile loss: $\mathcal{L} = \sum_q q\max(y-\hat{y}_q, 0) + (1-q)\max(\hat{y}_q-y, 0)$ |
| Supply elasticity | Estimated via natural experiments (price variation → driver response) |

**Key trick**: Airport demand is bimodal — high during flight arrivals, near-zero between. Standard models smooth this out. Solution: condition on flight schedule as known future covariate.

---

## Serving & Scale

```
Every 5 minutes per H3 hex:
  Feature assembly: historical demand + real-time signals + weather + events [2s] →
  TFT inference: predict demand for next 5/15/30/60 min [50ms per hex, batched] →
  Supply estimation: active drivers + expected arrivals [real-time from driver app] →
  Pricing optimizer: solve S(m) = D equilibrium [10ms] →
  Surge multiplier published to rider/driver apps →
  
Total: ~5s per pricing cycle (city-wide, all hexes in parallel)
```

- **Scale**: 10K+ H3 hexagons per major city, 600+ cities  
- **Real-time signals**: Kafka streams (app opens, GPS pings, trip completions)  
- **Geospatial**: H3 library for hex indexing, neighbor lookups  
- **Pricing guardrails**: Max surge cap (city-specific, typically 3–8×), min driver earnings floor  
- **Airport-specific**: Dedicated queue model (FIFO queue + demand matching)

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| MAPE (demand) | Forecast accuracy | <15% at 15-min horizon |
| Quantile coverage | Uncertainty calibration | P90 covers 90% of actuals |
| Surge accuracy | Predicted vs. actual imbalance | <20% error |
| ETA fulfillment | Rider wait time | P75 < 8 min |
| Driver utilization | Supply efficiency | >70% time occupied |
| Completed trip rate | Supply-demand match | >85% (requests → trips) |
| Revenue per trip | Business health | Stable or growing |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| ARIMA/Prophet | Simple, interpretable, good for trend | Can't handle complex spatial dependencies |
| DeepAR | Strong uncertainty quantification | Autoregressive = slow at long horizons |
| TFT (current) | Handles mixed inputs, attention, quantiles | Complex, expensive to train |
| Graph WaveNet | Captures spatial correlations | Hard to train, limited interpretability |
| Static pricing | Simple, predictable | Massive inefficiency (surplus/shortage) |
| Auction-based pricing | True market clearing | Bad UX (unpredictable prices) |
| Zone-based surge | Simpler than hex-level | Boundary effects, coarse |

---

## Recent Research (2024–2026)

1. **TimesFM** (Google, 2024) — Foundation model for time series (pre-trained on 100B time points)  
2. **Chronos** (Amazon, 2024) — Language model approach to time series forecasting  
3. **iTransformer** (2024) — Inverted Transformer with variate-token attention  
4. **PatchTST** (2024) — Patched time series Transformer (SOTA on long-horizon)  
5. **Dynamic pricing with fairness** (Uber, 2025) — Surge pricing with equity constraints (no price discrimination)  
6. **Spatial-temporal GNN for ride-hailing** (DiDi, 2024) — Hexagon-graph convolution for demand

---

## Advanced Trick Questions & Answers

**Q1: A major concert ends at 11 PM. Your model predicts 500 ride requests from that hexagon in the next 15 minutes. Actual demand is 2000. Why did it fail, and how do you fix it?**  
A: The event was likely not in training data (novel event) or the model under-weighted the event feature. Fixes: (1) Event magnitude feature: past events of similar size (stadium capacity × sold-out rate) as a covariate. (2) Causal uplift: "demand increase caused by event" = event_demand × decay_function(time_since_end). (3) Pre-positioning: move supply *before* event ends (predictable end time). (4) Human override: for mega-events (Super Bowl), manually set floor multiplier. The P90 quantile from TFT should have been closer — if not, your uncertainty is miscalibrated.

**Q2: Your surge pricing hits 3.5× at the airport. Riders complain on Twitter. The PR team demands you cap surge at 1.5×. What happens to the system?**  
A: With 1.5× cap and true equilibrium at 3.5×: (1) Demand remains high (riders see acceptable price), (2) Supply doesn't increase enough (drivers aren't incentivized to drive to airport), (3) Result: massive shortage → 30+ minute wait times → riders can't get rides at all → worse UX than high surge. (4) Counter-proposal: (a) cap at 2.5× with guaranteed ETA display, (b) pre-position drivers before peak (pay waiting bonus), (c) offer "wait and save" option (ride in 20 min at 1.5× vs. now at 3×).

**Q3: You forecast demand independently per hexagon. Hex A shows surplus, neighboring hex B shows shortage. What's wrong with independent forecasting?**  
A: Spatial correlation ignored. (1) Drivers in hex A will naturally flow to hex B (supply is mobile). (2) Riders in hex B might walk to hex A boundary. (3) Solution: spatial-aware model — either Graph Neural Network over hex adjacency, or post-processing: solve network flow optimization across hexes. (4) Joint optimization: surge in hex B attracts drivers from hex A → model this flow with supply elasticity that accounts for distance/time cost.

**Q4: Uber runs in 600+ cities. Training a TFT per city = 600 models. How do you scale?**  
A: (1) **Hierarchical/transfer**: one global model pre-trained on all cities, fine-tuned per city (transfer learning). Small cities benefit from large-city patterns. (2) **City embeddings**: add city_id as a learned embedding — single model handles all cities. (3) **Cluster cities**: group by characteristics (size, climate, transit availability) → one model per cluster. (4) **Foundation model approach** (TimesFM): pre-train on all Uber time series, zero-shot or few-shot per city. (5) In practice: ~10 model variants (by city tier) + city-specific feature engineering.

**Q5: Your model forecasts demand perfectly. But after deploying surge pricing, actual demand drops 30% vs. forecast. Your forecast is now "wrong." Is it actually wrong?**  
A: No — this is the **causal feedback loop** problem. Your forecast predicted *latent demand* (what would happen at base price). Surge pricing *changes* realized demand (price elasticity). (1) You need to forecast *pre-intervention* demand, then price optimally. (2) If you train on post-surge actuals, you learn "demand is low when surge is high" → model reduces surge → demand increases → model increases surge → oscillation. (3) Solution: train demand model on *intent signals* (app opens, searches) not completed trips. App opens happen before price is shown → uncontaminated signal.
