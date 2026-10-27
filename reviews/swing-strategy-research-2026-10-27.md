# What a good swing strategy looks like — evidence review (2026-10-27)

Reference for DeepSwing's rules. It ranks the strategy families by how well the evidence survives the things that broke DeepSwing: out-of-sample periods, trading costs, survivorship bias and small accounts. It also says which ones fit a 1–8 week holding period on Nordic and US stocks. Every candidate must also clear this repo's own bar (`reviews/backtest-2026-10-27.md`): improve both Nordic and US, in both 2016–22 and 2023–26, and beat simply holding the same liquid stocks equal-weight.

## The base rates to keep in mind

- **Published edges shrink.** Across 97 published return predictors, returns were 26% lower out-of-sample and 58% lower after publication (McLean & Pontiff, *JF* 2016). The strongest in-sample results decay the most. Expect any textbook rule to be worth roughly half of what its backtest says.
- **Most active traders lose.** 97% of Brazilian futures day traders who persisted for 300+ days lost money (Chague, De-Losso & Giovannetti 2020). In Taiwan, under 1% of day traders were reliably profitable net of fees (Barber, Lee, Liu & Odean). Swing horizons are less cost-dominated, but the lesson is the same: costs and turnover decide most outcomes.
- **Free data flatters long-only stock backtests.** Yahoo omits delisted names, which typically overstates returns by about 1–4% a year. Dip-buying rules are hurt more than trend rules, because the missing stocks are the ones whose dips never recovered. Backtesting only current index members overstated one S&P 1500 pullback system by about 2 points of CAGR (StockCharts).

## Strategy families, ranked by evidence for a swing horizon

| # | Family | Evidence | Typical horizon | Fit for DeepSwing |
|---|---|---|---|---|
| 1 | **Opportunistic / cluster insider buying** | Strong academic (US) + consistent Swedish theses | 1–6 months | **Best fit** — data available, an LLM helps |
| 2 | **Short-term momentum in liquid, high-turnover stocks** | Peer-reviewed (*RFS* 2022), survives costs, largest stocks | ~1 month | Good — testable now |
| 3 | **Trend following: new highs + wide ATR trailing stop** | Strong, survivorship-free, 1950–2024 | Months–years | Partial — works, but not as a swing trade |
| 4 | **Intermediate momentum / 52-week-high proximity** | Strong academic; Nordic theses agree | 3–12 months, monthly rebalance | Partial — our quick test was erratic |
| 5 | Short-term reversal / RSI(2)-style dip buying | Classic effect largely gone; works mainly on indices/ETFs and large caps | Days–1 week | Weak for single stocks |
| 6 | Post-earnings-announcement drift | Largely arbitraged away in large caps | Weeks | Low |
| 7 | Chart patterns, indicator combinations, "setups" | Little rigorous evidence | Days–weeks | Avoid as a primary edge |

### 1. Insider buying — opportunistic and clustered

- Cohen, Malloy & Pomorski (*JF* 2012) split insider trades into **routine** ones (same calendar month for several years running, about 55% of trades) and **opportunistic** ones. Following only opportunistic trades earned about **82 bp/month** of abnormal return; routine trades earned about zero. The signal was strongest for non-executive insiders and poorly governed firms. Sample: US, 1986–2007.
- Sweden: master's-thesis evidence (Lund, Gothenburg, Aalto, BI) consistently finds positive abnormal returns after insider *purchases*. Sales carry little signal. The effect is strongest in **small and mid caps**, and **clusters** (several insiders buying within a short window) are the strongest signal (125,657 Stockholm trades, 2005–2014).
- Why it suits this project: it is information the price chart does not show, and it is slow-moving (holds of weeks to months). Telling routine from opportunistic, or a share-programme exercise from an open-market buy, is exactly the reading task an LLM is good at.
- Data: FI's search export (`marknadssok.fi.se/Publiceringsklient/sv-SE/Search/Search?SearchFunctionType=Insyn&…&button=export`) returns historical transactions as CSV (checked back to 2017). SEC Form 4 covers the US. **Note:** the CSV URL that `financedata.insider` uses (`fi.se/contentassets/…/insynsregistret.csv`) now returns 404, so the live Swedish insider feed is probably empty.

### 2. Short-term momentum (high-turnover stocks)

- Medhat & Schmeling (*RFS* 2022, US 1963–2018 plus international) sort on last month's return and share turnover. Low-turnover stocks show the familiar one-month **reversal**; **high-turnover stocks show one-month momentum**. The momentum survives costs, is as profitable as conventional momentum with less crash risk, and is **strongest in the largest, most liquid stocks**.
- Fits a swing horizon (hold about a month) and our liquid universe. Rules to test: monthly, rank liquid names by last-month return among the top third by turnover (volume ÷ shares or ÷ average volume), buy the top N, hold one month.

### 3. Trend following on single stocks

- Wilcox & Crittenden (2005); Zarattini, Pagani & Wilcox (2025): survivorship-free US data 1950–2024, 66,000+ trades.
  - **Entry:** close at a new all-time high (adjusted), buy at the next open.
  - **Exit:** trailing stop at `ATH × (1 − 10 × ATR42 / close)` — roughly **10 ATRs**, never lowered; sell at the next open.
  - **Universe:** price above $10, 42-day dollar volume above $1M.
  - **Results:** 1991–2024 CAGR 15.0% vs 11.2% for the market; max drawdown 32% vs 55%. Win rate 43.9%, average win **+1.90R** vs average loss −0.70R, winners held about **370 days**. Fewer than 7% of trades produce most of the profit. Stable out-of-sample over 2005–24.
- **The catch for a small account:** with realistic costs, a $100k portfolio made **2.45% CAGR** (alpha −4.6%), because frequent small rebalances are eaten by commissions. A turnover-control layer brought it back to about 13% (it skips small rebalances and trades when costs are low relative to size). The portfolio also ran about 120% average leverage.
- Lesson for DeepSwing: the evidence for "let winners run" is real, but it is a **position-trading** horizon (months), not two-week swings. Our own test agrees: removing the fixed target and widening the trail helped Nordic but not US at swing horizons.

### 4. Intermediate momentum and the 52-week high

- 12-1 month momentum and nearness to the 52-week high (George & Hwang, *JF* 2004) are among the most replicated anomalies. The 52-week-high version subsumes much of plain momentum, shows little long-run reversal, and was positive in 17 of 18 international markets in one replication. Nordic theses (CBS, Lund, Vaasa) find momentum profits in Sweden, Finland, Denmark and Norway, stronger in small firms and weaker than in the US.
- Known risks: momentum crashes, and post-publication decay. Our own quick test (top-20/50, monthly) beat equal-weight in some periods, but the US excess came almost entirely from 2020 and drawdowns were deeper.

### 5. Short-term reversal, RSI(2), IBS

- The one-week/one-month reversal is real in history, but most of it sat inside the bid-ask spread for small caps. After costs it survives only in **large caps** (Quantpedia summary of de Groot, Huij & Zhou: +29 bp/week net for the top 100 US stocks, negative for a 1,500-stock universe). Blitz et al. report that the classic effect "has largely vanished in most regions", while residual or industry-adjusted versions persist (*JFM* 2013; Robeco 2023).
- RSI(2) (Connors & Alvarez 2008) and IBS dip-buying are documented mainly on **indices and ETFs**; claims of persistence come from vendors. On single stocks, survivorship bias specifically flatters dip-buying.
- Use: possibly as an *entry-timing* refinement inside an uptrend (buy the pullback rather than the breakout), not as an edge in itself. Our own test: the RSI/volume pullback filter added nothing beyond the trend filter.

### 6. Post-earnings-announcement drift

- Martineau (2021, "Rest in Peace Post-Earnings Announcement Drift") and others find prices now absorb earnings news on the day, with no drift in recent large-cap data. What remains is small and in less-covered small caps. Low priority, except possibly for under-covered Nordic small caps, where Swedish-language reports matter.

## What the evidence agrees a good swing system has

1. **A liquid universe.** Most anomaly profits in small or illiquid names vanish after costs, and gaps through stops cluster there. (Already done: 10 MSEK/day, 10 SEK price floor.)
2. **Trades in the direction of the intermediate trend** (price above long averages, near highs). Dip-buying without a trend filter is the weakest family. (Done.)
3. **Volatility-scaled exits wide enough to sit outside daily noise.** Every serious study uses ATR- or volatility-based stops, and the trend studies use wide ones (about 10 ATR). (Partly done: 1.5–2.5×ATR; our tests say tighter is always worse.)
4. **Positive skew:** win rate around 35–45%, average win well above average loss, with a few outliers paying for everything. A fixed near target caps exactly the trades that matter. (Not done: the fixed `min_rrr` target remains.)
5. **Low turnover and cost awareness.** For a 100k account, costs can turn a 15% strategy into 2.5%. Fewer, longer trades; no re-entries for tiny adjustments.
6. **An information edge, not only a price pattern.** Price-only rules on daily bars are the most crowded and most decayed. The persistent edges in the literature involve information that is slow to diffuse: insider purchases, attention/turnover, under-covered firms.
7. **Honest validation:** survivorship-aware data where possible, out-of-sample periods, a buy-and-hold benchmark, and an expectation that live results come in at about half of the backtest.

## Reference rule sets to test next (pre-registered)

Each is judged against the same bar: positive in Nordic and US (or Sweden only for A), in 2016–22 *and* 2023–26, and ahead of equal-weight holding.

| Test | Rules | Data | Effort |
|---|---|---|---|
| **A. Swedish insider clusters** | Buy at the next open after ≥2 distinct insiders make open-market purchases (not share-programme exercises) within 30 days, total ≥ 500k SEK; hold 60 trading days or until a 10×ATR trail; liquid-universe filter | FI export 2016–2026 + cached OHLCV | Medium: new data loader + event backtest |
| **B. Short-term momentum** | Monthly: within the liquid universe, take the top third by 1-month turnover, rank by 1-month return, buy the top 20 equal-weight, hold 1 month | Cached OHLCV | Low: extends `momentum.py` |
| **C. Position-trend variant** | Current entry gates plus a 52-week high within 5%; no fixed target; trail `high × (1 − k·ATR/close)`, k ∈ {6, 10}; minimum hold before rebalancing | Cached OHLCV | Low: engine settings + one exit mode |
| **D. Model as a filter, not a picker** | Run the best of A–C mechanically; the LLM may only veto (e.g. for earnings within days, scandal, a routine insider trade); judge vetoes on replayed history | Replay harness | Medium |

## Sources

- McLean, R. D. & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*. [PDF](https://ivey.uwo.ca/media/3775549/pontiff.pdf)
- Cohen, L., Malloy, C. & Pomorski, L. (2012). Decoding Inside Information. *Journal of Finance*. [NBER](https://www.nber.org/papers/w16454) · [HBS summary](https://hbswk.hbs.edu/item/decoding-inside-information)
- Swedish insider-trading theses: [Aalto](https://aaltodoc.aalto.fi/items/46b03acc-1857-4858-baf5-5e353e2057e6), [Lund](https://www.lunduniversity.lu.se/lup/publication/7793807), [Lund 2026](https://lup.lub.lu.se/student-papers/record/9246264), [Gothenburg](https://gupea.ub.gu.se/handle/2077/88201), [Gothenburg (large vs small cap)](https://gupea.ub.gu.se/items/d213ae81-4b0c-4240-a522-e913d255bc93), [BI](https://biopen.bi.no/bi-xmlui/handle/11250/2579644)
- Medhat, M. & Schmeling, M. (2022). Short-term Momentum. *Review of Financial Studies*. [Accepted manuscript](https://openaccess.city.ac.uk/id/eprint/31278/) · [Alpha Architect summary](https://alphaarchitect.com/short-term-momentum/)
- Zarattini, C., Pagani, A. & Wilcox, C. (2025). Does Trend Following Still Work on Stocks? [Concretum](https://concretumgroup.com/does-trend-following-still-work-on-stocks/) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5084316) · Wilcox & Crittenden (2005) [PDF](https://www.cis.upenn.edu/~mkearns/finread/trend.pdf)
- George, T. & Hwang, C.-Y. (2004). The 52-Week High and Momentum Investing. *Journal of Finance*. [Alpha Architect summary](https://alphaarchitect.com/the-secret-to-momentum-is-the-52-week-high/) · Liu, Liu & Ma (2011) international replication, [JIMF](https://ideas.repec.org/a/eee/jimfin/v30y2011i1p180-204.html)
- Nordic momentum theses: [CBS](https://research.cbs.dk/en/studentProjects/momentum-investment-strategies-a-study-of-momentum-returns-and-th/), [Lund](https://www.lunduniversity.lu.se/lup/publication/1348919), [Vaasa](https://osuva.uwasa.fi/handle/11111/11305)
- Short-term reversal: [Quantpedia](https://quantpedia.com/strategies/short-term-reversal-in-stocks) · Blitz, Huij, Lansdorp & Verbeek (2013), [Short-term residual reversal](https://repub.eur.nl/pub/40707) · [Robeco 2023](https://www.robeco.com/en-latam/insights/2023/10/reversing-the-trend-of-short-term-reversal) · [EFMA 2011 paper](https://efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2011-Braga/papers/0259_update.pdf)
- RSI(2)/IBS practitioner material: [LuxAlgo RSI-2](https://www.luxalgo.com/library/concept/rsi-2/), [StockCharts SystemTrader](https://articles.stockcharts.com/article/articles-arthurhill-2016-12-systemtrader---testing-and-tweaking-an-rsi-mean-reversion-system-for-spy-qqq-and-ijr/), [Quantified Strategies IBS](https://quantifiedstrategies.substack.com/p/the-classical-mean-reversion-strategy)
- Pullback/trend filters and survivorship: [StockCharts (Arthur Hill)](https://articles.stockcharts.com/article/articles-arthurhill-2016-06-systemtrader---testing-macd-signal-line-crosses---reducing-drawdowns-with-simple-trend-rule---increasing-gains-with-pullback-rule/), [LuxAlgo on survivorship bias](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained)
- PEAD decline: [Martineau (2021)](https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf), [Quantpedia: 50 years of PEAD](https://quantpedia.com/Blog/Details/50-years-in-pead-research), [Columbia: PEAD declined over time](https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf)
- Retail trader outcomes: [CNBC on day-trader studies](https://www.cnbc.com/2020/11/20/attention-robinhood-power-users-most-day-traders-lose-money.html)
- FI insider register tooling: [insynsregistret (GitHub)](https://github.com/w3stling/insynsregistret)
