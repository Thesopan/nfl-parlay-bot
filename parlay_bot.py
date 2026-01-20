# pip install --upgrade pip
# pip install nflreadpy pandas numpy scipy polars pyarrow
# (fallback) pip install nfl-data-py
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Prefer nflreadpy; fallback to nfl_data_py
try:
    import nflreadpy as nfl
    NFL_BACKEND = "nflreadpy"
except Exception:
    import nfl_data_py as nfl  # type: ignore
    NFL_BACKEND = "nfl_data_py"

CURRENT_SEASON = 2025

@dataclass
class Leg:
    player: str
    stat: str      # pass_yds | rush_yds | rec_yds | receptions | pass_td | rush_td | rec_td
    threshold: float
    opponent: Optional[str] = None  # e.g., "PHI"

# Column aliases seen across sources
STAT_ALIASES = {
    "pass_yds": ["passing_yards", "pass_yds", "pass_yards"],
    "rush_yds": ["rushing_yards", "rush_yds", "rush_yards"],
    "rec_yds":  ["receiving_yards", "rec_yds", "rec_yards"],
    "receptions": ["receptions", "rec"],
    "pass_td":    ["passing_tds", "pass_td", "passing_touchdowns"],
    "rush_td":    ["rushing_tds", "rush_td", "rushing_touchdowns"],
    "rec_td":     ["receiving_tds", "rec_td", "receiving_touchdowns"],
}

# Which positions should count for each stat (for filtering + defense allowances)
POS_FILTER = {
    "pass_yds": ["QB"], "pass_td": ["QB"],
    "rec_yds": ["WR", "TE", "RB"], "receptions": ["WR", "TE", "RB"], "rec_td": ["WR", "TE", "RB"],
    "rush_yds": ["RB", "QB", "WR"], "rush_td": ["RB", "QB", "WR"],
}

TEAM_ABBR_FIX = {"LA": "LAR", "WSH": "WAS", "WFT": "WAS", "JAX": "JAX", "LV": "LV"}

# ----------------- Data loading / normalization -----------------
def _load_weekly(seasons: List[int]) -> pd.DataFrame:
    if NFL_BACKEND == "nflreadpy":
        df = nfl.load_player_stats(seasons)
        try:
            import polars as pl  # type: ignore
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except Exception:
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            else:
                df = pd.DataFrame(df)
    else:
        # nfl_data_py
        try:
            df = nfl.import_weekly_data(seasons, downcast=True)
        except TypeError:
            df = nfl.import_weekly_data(seasons, None)

    df.columns = [c.lower() for c in df.columns]
    for c in ("recent_team", "team", "posteam"):
        if c in df.columns:
            df = df.rename(columns={c: "team"}); break
    for c in ("opponent_team", "opp", "defteam"):
        if c in df.columns:
            df = df.rename(columns={c: "opp"}); break
    for c in ("player_name", "name", "player"):
        if c in df.columns:
            df = df.rename(columns={c: "player_name"}); break
    return df

def _extract_stat_column(df: pd.DataFrame, key: str) -> str:
    for cand in STAT_ALIASES.get(key, []):
        if cand in df.columns:
            return cand
    raise KeyError(f"Stat '{key}' not found. Columns sample: {list(df.columns)[:20]}")

def _abbr_fix(abbr: Optional[str]) -> Optional[str]:
    if abbr is None: return None
    return TEAM_ABBR_FIX.get(abbr, abbr)

def _defense_allowance_table(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-opponent means for each stat, filtered to relevant positions."""
    out: Dict[str, Dict[str, float]] = {}
    if "opp" not in df.columns:
        return out
    for stat_key, aliases in STAT_ALIASES.items():
        col = next((c for c in aliases if c in df.columns), None)
        if not col: 
            continue
        d = df
        if "position" in d.columns and stat_key in POS_FILTER:
            d = d[d["position"].isin(POS_FILTER[stat_key])]
        means = d.groupby("opp")[col].mean(numeric_only=True)
        for t, v in means.items():
            out.setdefault(t, {})[stat_key] = float(v)
    return out

# ----------------- Prob helpers -----------------
def _empirical_prob(x: float, samples: np.ndarray, tau: float = 6.0) -> float:
    """Exponentially-weighted empirical P(sample >= x), heavier weight on recent games."""
    n = len(samples)
    if n == 0: return float("nan")
    w = np.exp(-np.linspace(n-1, 0, n) / tau)
    w = w / w.sum()
    return float((w * (samples >= x)).sum())

def _poisson_tail(k: float, lam: float) -> float:
    """P(X >= k) for Poisson(lam)."""
    kk = int(math.ceil(k))
    if lam <= 0: return 1.0 if kk <= 0 else 0.0
    s = 0.0
    for i in range(kk):
        s += (lam ** i) * math.exp(-lam) / math.factorial(i)
    return max(0.0, 1.0 - s)

def _normal_tail(thresh: float, mu: float, sigma: float) -> float:
    """P(X >= thresh) for Normal(mu, sigma)."""
    if sigma <= 1e-9: 
        return 1.0 if thresh <= mu else 0.0
    z = (thresh - mu) / sigma
    return float(1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

# ----------------- Player matching -----------------
def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def _find_player_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """full name → initials+last → last only; then keep the most frequent player_id."""
    if "player_name" not in df.columns:
        return df.iloc[0:0]
    names = df["player_name"].fillna("").astype(str)

    def norm(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

    q_raw = query.strip()
    toks = [t for t in re.split(r"\s+", q_raw) if t]
    last = toks[-1] if toks else q_raw
    initials = "".join(t[0] for t in toks[:-1]) if len(toks) >= 2 else ""
    q_norm = norm(q_raw)

    # 1) normalized containment
    hits = df[names.map(norm).str.contains(q_norm, na=False)]
    # 2) initials+last
    if hits.empty and initials and last:
        pat_init = r"\.? *".join(list(initials))
        pat = rf"{pat_init}\.? *{re.escape(last)}"
        try:
            hits = df[names.str.contains(pat, case=False, regex=True, na=False)]
        except re.error:
            hits = df[names.str.contains(last, case=False, na=False)]
    # 3) last only
    if hits.empty and last:
        try:
            pat_last = re.compile(rf"(?:^|[^A-Za-z]){re.escape(last)}(?:[^A-Za-z]|$)", re.IGNORECASE)
            hits = df[names.str.contains(pat_last, na=False)]
        except re.error:
            hits = df[names.str.contains(last, case=False, na=False)]
    if hits.empty:
        return hits

    pid_col = "player_id" if "player_id" in hits.columns else None
    if pid_col:
        counts = hits.groupby(pid_col).size().sort_values(ascending=False)
        hits = hits[hits[pid_col] == counts.index[0]]
    else:
        top_name = hits["player_name"].value_counts().index[0]
        hits = hits[hits["player_name"] == top_name]
    return hits

# ----------------- Core estimator -----------------
def estimate_leg_prob(
    df_weekly: pd.DataFrame,
    defense_table: Dict[str, Dict[str, float]],
    leg: Leg,
    lookback_games: int = 12,
    blend: float = 0.7,
) -> Tuple[float, Dict[str, float]]:
    stat_col = _extract_stat_column(df_weekly, leg.stat)
    # player rows
    df_p = _find_player_rows(df_weekly, leg.player).copy()
    if "position" in df_p.columns and leg.stat in POS_FILTER:
        df_p = df_p[df_p["position"].isin(POS_FILTER[leg.stat])]
    if df_p.empty:
        return (float("nan"), {"error": f"player not found (after position filter): {leg.player}"})

    # CURRENT SEASON ONLY for player form
    if "season" in df_p.columns:
        df_p = df_p[df_p["season"] == CURRENT_SEASON]

    sort_cols = [c for c in ("season", "week") if c in df_p.columns]
    if sort_cols: df_p = df_p.sort_values(sort_cols)

    series = df_p[stat_col].dropna().astype(float).tail(lookback_games)
    if series.empty:
        return (float("nan"), {"error": "no recent games for this stat"})

    mu_player = float(series.mean())
    sd_player = float(series.std(ddof=1)) if len(series) > 1 else max(1.0, mu_player * 0.25)

    opp = _abbr_fix(leg.opponent) if leg.opponent else None
    if opp and opp in defense_table and leg.stat in defense_table[opp]:
        mu_def = float(defense_table[opp][leg.stat])
    else:
        # global mean for this stat (position-filtered)
        d = df_weekly
        if "position" in d.columns and leg.stat in POS_FILTER:
            d = d[d["position"].isin(POS_FILTER[leg.stat])]
        mu_def = float(d[stat_col].dropna().astype(float).mean())

    mu_adj = blend * mu_player + (1.0 - blend) * mu_def

    # empirical (EW) + model tail
    p_emp = _empirical_prob(leg.threshold, series.values, tau=6.0)
    if leg.stat in ("pass_td", "rush_td", "rec_td"):
        p_model = _poisson_tail(leg.threshold, max(0.01, mu_adj))
    else:
        p_model = _normal_tail(leg.threshold, mu_adj, sd_player)

    alpha = 0.75  # lean more on recent form
    if p_emp is None or (isinstance(p_emp, float) and math.isnan(p_emp)):
        p = p_model
    else:
        p = alpha * p_emp + (1 - alpha) * p_model

    return p, {
        "player_mean": mu_player,
        "player_sd": sd_player,
        "def_allow_mean": mu_def,
        "blend": blend,
        "mu_adj": mu_adj,
        "p_emp": p_emp,
        "p_model": p_model,
        "backend": NFL_BACKEND,
    }

import os, json, hashlib

def _seasons_key(seasons: List[int]) -> str:
    s = json.dumps(sorted(seasons))
    return hashlib.md5(s.encode()).hexdigest()[:10]

def _load_weekly_cached(seasons: List[int]) -> pd.DataFrame:
    os.makedirs("data_cache", exist_ok=True)
    key = _seasons_key(seasons)
    path = os.path.join("data_cache", f"weekly_{key}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = _load_weekly(seasons)
    df.to_parquet(path, index=False)
    return df

def build_model(seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    weekly = _load_weekly_cached(seasons)
    defense = _defense_allowance_table(weekly)
    return weekly, defense

def score_parlay(legs: List[Leg], weekly: pd.DataFrame, defense: Dict[str, Dict[str, float]],
                 lookback_games: int = 12, blend: float = 0.7) -> pd.DataFrame:
    rows = []
    for lg in legs:
        p, info = estimate_leg_prob(weekly, defense, lg, lookback_games, blend)
        rows.append({"player": lg.player, "stat": lg.stat, "threshold": lg.threshold,
                     "opponent": lg.opponent, "prob": p, **info})
    out = pd.DataFrame(rows)
    out["prob_pct"] = (out["prob"] * 100).round(1)
    want = ["player", "stat", "threshold", "opponent", "prob_pct",
            "player_mean", "def_allow_mean", "mu_adj", "backend"]
    for c in want:
        if c not in out.columns:
            out[c] = np.nan
    return out[want]

# ----------------- CLI -----------------
if __name__ == "__main__":
    # current season only for model + defense table
    weekly_df, def_tbl = build_model([CURRENT_SEASON])

    print("Enter parlay legs (blank player to finish).")
    print("Stat types: pass_yds | rush_yds | rec_yds | receptions | pass_td | rush_td | rec_td")

    legs: List[Leg] = []
    while True:
        player = input("Player name (blank to run): ").strip()
        if player == "": break
        stat = input("Stat type: ").strip()
        if stat not in ("pass_yds","rush_yds","rec_yds","receptions","pass_td","rush_td","rec_td"):
            print("Invalid stat type. Try again."); continue
        th_s = input("Threshold (e.g., 200 for yards, 7 for receptions, 2 for TDs): ").strip()
        try:
            threshold = float(th_s)
        except ValueError:
            print("Threshold must be a number. Try again."); continue
        opp = input("Opponent (abbr like PHI, empty if unknown): ").strip().upper() or None
        legs.append(Leg(player=player, stat=stat, threshold=threshold, opponent=opp))

    if not legs:
        print("No legs entered. Exiting.")
    else:
        df = score_parlay(legs, weekly_df, def_tbl, lookback_games=12, blend=0.7)
        for _, r in df.iterrows():
            player = str(r.get("player", "")).strip()
            stat = str(r.get("stat", "")).strip()
            thr = r.get("threshold", "")
            pct = r.get("prob_pct", float("nan"))
            print(f"{player}  {stat} {thr}+  :  {'N/A' if pd.isna(pct) else f'{pct:.1f}%'}")

        valid = df["prob_pct"].dropna() / 100.0
        overall = float(np.prod(valid)) if len(valid) else float("nan")
        print(f"Parlay hit probability ≈ {overall*100:.1f}% (assumes independence)" if not math.isnan(overall)
              else "Parlay hit probability: N/A (one or more legs missing data)")

#& .\.venv\Scripts\Activate.ps1; python .\parlay_bot.py
