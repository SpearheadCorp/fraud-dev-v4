"""
Pod 1: Data Gather (GPU)
Generates synthetic credit card transaction data on GPU using cuDF/CuPy.
Statistical distributions fitted from real seed data (scipy, one-time CPU).
All random generation + DataFrame construction + parquet writes on GPU.
Supports 'once' and 'continuous' run modes.
"""
import os
import sys
import time
import signal
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    log.info("Signal %s received — shutting down gracefully", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/raw"))
HOSTNAME = os.environ.get("HOSTNAME", f"gather-{os.getpid()}")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000000"))
FRAUD_RATE = float(os.environ.get("FRAUD_RATE", "0.005"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "1000000"))
RUN_MODE = os.environ.get("RUN_MODE", "once")
KAGGLE_SEED_PATH = os.environ.get("KAGGLE_SEED_PATH", "")
TARGET_ROWS_PER_SEC = int(os.environ.get("TARGET_ROWS_PER_SEC", "0"))

# Identity pool sizes — shared across chunks so the GNN graph has meaningful
# connectivity (same card → multiple transaction nodes).
NUM_USERS = 10_000
NUM_MERCHANTS = 1_000

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
    "health_fitness", "travel", "kids_pets", "home",
]
HIGH_FRAUD_CATEGORIES = {"shopping_net", "travel", "misc_net"}
HIGH_FRAUD_INDICES = {i for i, c in enumerate(ALL_CATEGORIES) if c in HIGH_FRAUD_CATEGORIES}

CATEGORY_MAX_AMT = {
    "grocery_pos":    500.0,
    "grocery_net":    500.0,
    "gas_transport":  200.0,
    "food_dining":    300.0,
    "personal_care":  150.0,
    "kids_pets":      500.0,
    "health_fitness": 300.0,
    "entertainment":  500.0,
    "home":          5000.0,
    "travel":       10000.0,
    "shopping_pos":  5000.0,
    "shopping_net":  5000.0,
    "misc_pos":      3000.0,
    "misc_net":      3000.0,
}
CATEGORY_MIN_AMT = {
    "grocery_pos":     8.0,
    "grocery_net":    15.0,
    "gas_transport":   5.0,
    "food_dining":     4.0,
    "personal_care":   5.0,
    "kids_pets":       8.0,
    "health_fitness": 15.0,
    "entertainment":   5.0,
    "home":           20.0,
    "travel":         89.0,
    "shopping_pos":    8.0,
    "shopping_net":    5.0,
    "misc_pos":        3.0,
    "misc_net":        5.0,
}
# Realistic merchant name pools keyed by category.
# Names use the "fraud_" prefix to match the Kaggle-style synthetic data format.
CATEGORY_MERCHANTS = {
    "grocery_pos": [
        "fraud_Whole_Foods_Market", "fraud_Kroger_Co", "fraud_Safeway_Inc",
        "fraud_Publix_Super_Markets", "fraud_Trader_Joes", "fraud_Aldi_Inc",
        "fraud_Costco_Wholesale", "fraud_Walmart_Grocery", "fraud_Target_Grocery",
        "fraud_Wegmans_Food", "fraud_HEB_Grocery", "fraud_Meijer_Inc",
        "fraud_Stop_Shop", "fraud_Food_Lion_LLC", "fraud_Giant_Food",
        "fraud_Sprouts_Farmers", "fraud_Winn_Dixie", "fraud_Harris_Teeter",
        "fraud_ShopRite_Stores", "fraud_Ralphs_Grocery",
    ],
    "grocery_net": [
        "fraud_Instacart_Inc", "fraud_Amazon_Fresh", "fraud_Walmart_Online_Grocery",
        "fraud_Target_Drive_Up", "fraud_Shipt_Delivery", "fraud_FreshDirect_LLC",
        "fraud_Peapod_Online", "fraud_Thrive_Market", "fraud_Boxed_Wholesale",
        "fraud_Imperfect_Foods", "fraud_Misfits_Market", "fraud_Good_Eggs_Inc",
        "fraud_Hungryroot_Inc",
    ],
    "gas_transport": [
        "fraud_Shell_Oil_Co", "fraud_ExxonMobil_Corp", "fraud_BP_Gas_Station",
        "fraud_Chevron_Corp", "fraud_Valero_Energy", "fraud_Sunoco_LP",
        "fraud_Marathon_Petroleum", "fraud_Phillips_66", "fraud_76_Gas_Station",
        "fraud_Speedway_LLC", "fraud_Circle_K_Stores", "fraud_Wawa_Fuel",
        "fraud_QuikTrip_Corp", "fraud_Caseys_General", "fraud_RaceTrac_Petroleum",
        "fraud_Pilot_Flying_J", "fraud_Loves_Travel", "fraud_Uber_Rideshare",
        "fraud_Lyft_Inc", "fraud_MTA_Transit_Auth",
    ],
    "food_dining": [
        "fraud_McDonalds_Corp", "fraud_Starbucks_Coffee", "fraud_Chipotle_Mexican",
        "fraud_Subway_Restaurants", "fraud_Dominos_Pizza", "fraud_Pizza_Hut_LLC",
        "fraud_Olive_Garden_Rest", "fraud_Panera_Bread_Co", "fraud_Chilis_Grill",
        "fraud_Applebees_Intl", "fraud_Dunkin_Brands", "fraud_Burger_King_Corp",
        "fraud_Taco_Bell_Corp", "fraud_Wendys_Intl", "fraud_KFC_Corp",
        "fraud_IHOP_Restaurants", "fraud_Dennys_Inc", "fraud_Cracker_Barrel",
        "fraud_Buffalo_Wild_Wings", "fraud_Cheesecake_Factory",
    ],
    "entertainment": [
        "fraud_Netflix_Inc", "fraud_Spotify_AB", "fraud_AMC_Theatres",
        "fraud_Regal_Entertainment", "fraud_Hulu_LLC", "fraud_Disney_Plus",
        "fraud_Xbox_Live_Gold", "fraud_PlayStation_Network", "fraud_Steam_Games",
        "fraud_Ticketmaster_Inc", "fraud_StubHub_Inc", "fraud_Apple_iTunes",
        "fraud_YouTube_Premium", "fraud_HBO_Max_Subs", "fraud_Peacock_TV",
        "fraud_Amazon_Prime_Video", "fraud_Twitch_Interactive", "fraud_ESPN_Plus",
        "fraud_Paramount_Plus", "fraud_Discovery_Plus",
    ],
    "travel": [
        "fraud_Delta_Air_Lines", "fraud_United_Airlines", "fraud_American_Airlines",
        "fraud_Southwest_Airlines", "fraud_Marriott_Hotels", "fraud_Hilton_Worldwide",
        "fraud_Airbnb_Inc", "fraud_Hyatt_Hotels_Corp", "fraud_Expedia_Group",
        "fraud_Booking_Holdings", "fraud_Choice_Hotels_Intl", "fraud_IHG_Hotels",
        "fraud_Wyndham_Hotels", "fraud_Enterprise_Rent_Car", "fraud_Hertz_Corp",
        "fraud_Avis_Budget_Group", "fraud_Royal_Caribbean_Intl", "fraud_Carnival_Cruise",
        "fraud_Norwegian_Cruise", "fraud_Amtrak_Rail",
    ],
    "shopping_pos": [
        "fraud_Macys_Inc", "fraud_Nordstrom_Inc", "fraud_TJ_Maxx_Corp",
        "fraud_Best_Buy_Co", "fraud_Gap_Inc", "fraud_Old_Navy_LLC",
        "fraud_Home_Depot_Inc", "fraud_Lowes_Companies", "fraud_Target_Corp",
        "fraud_Kohls_Corp", "fraud_Bed_Bath_Beyond", "fraud_Ross_Stores_Inc",
        "fraud_Burlington_Coat", "fraud_DSW_Shoes_Inc", "fraud_Foot_Locker_Inc",
        "fraud_GameStop_Corp", "fraud_Five_Below_Inc", "fraud_Dollar_General",
        "fraud_Marshalls_Inc", "fraud_Sears_Holdings",
    ],
    "shopping_net": [
        "fraud_Amazon_Marketplace", "fraud_eBay_Inc", "fraud_Etsy_Inc",
        "fraud_Wayfair_LLC", "fraud_Chewy_Inc", "fraud_Zara_Online",
        "fraud_ASOS_PLC", "fraud_Shein_Group", "fraud_Zappos_Amazon",
        "fraud_Overstock_com", "fraud_Newegg_Commerce", "fraud_Walmart_Online",
        "fraud_Target_Online_Retail", "fraud_Rakuten_Inc", "fraud_Wish_Inc",
        "fraud_Nike_Direct_Store", "fraud_Adidas_Online", "fraud_Apple_Store_Online",
        "fraud_Microsoft_Store", "fraud_Shopify_Merchant",
    ],
    "personal_care": [
        "fraud_Ulta_Beauty_Inc", "fraud_Sephora_Inc", "fraud_CVS_Pharmacy",
        "fraud_Walgreens_Boots", "fraud_Great_Clips_Inc", "fraud_Sport_Clips_Inc",
        "fraud_Regis_Salons", "fraud_Sally_Beauty_Co", "fraud_Bath_Body_Works",
        "fraud_Aveda_Corp", "fraud_Supercuts_Inc", "fraud_Hair_Cuttery",
        "fraud_Rite_Aid_Corp", "fraud_Cost_Cutters", "fraud_OPI_Products",
        "fraud_Estee_Lauder_Co", "fraud_LUSH_Cosmetics", "fraud_Nail_Pro_Salon",
    ],
    "health_fitness": [
        "fraud_Planet_Fitness_Inc", "fraud_LA_Fitness_Intl", "fraud_Golds_Gym",
        "fraud_Equinox_Holdings", "fraud_CVS_Health_Corp", "fraud_Walgreens_Health",
        "fraud_GNC_Holdings", "fraud_Vitamin_Shoppe_Inc", "fraud_Peloton_Interactive",
        "fraud_ClassPass_Inc", "fraud_Anytime_Fitness", "fraud_24_Hour_Fitness",
        "fraud_YMCA_USA", "fraud_SoulCycle_Inc", "fraud_Orangetheory_Fitness",
        "fraud_Crunch_Fitness", "fraud_Bowflex_Direct", "fraud_MyFitnessPal_Inc",
    ],
    "kids_pets": [
        "fraud_PetSmart_Inc", "fraud_Petco_Animal_Supplies", "fraud_Chewy_Pets",
        "fraud_Build_A_Bear_Workshop", "fraud_Chuck_E_Cheese", "fraud_Kids_Foot_Locker",
        "fraud_Carters_Inc", "fraud_OshKosh_BGosh", "fraud_Petsense_LLC",
        "fraud_Learning_Express", "fraud_Gymboree_Corp", "fraud_Justice_Girls",
        "fraud_American_Girl_LLC", "fraud_Stride_Rite_Corp", "fraud_PetMeds_Express",
        "fraud_BarkBox_Subscription", "fraud_Legoland_Parks", "fraud_Discovery_Kids",
    ],
    "home": [
        "fraud_IKEA_North_America", "fraud_Home_Depot_Inc", "fraud_Lowes_Home_Imp",
        "fraud_Wayfair_Home_Decor", "fraud_Restoration_Hardware", "fraud_Williams_Sonoma",
        "fraud_Pottery_Barn_LLC", "fraud_Crate_Barrel_Inc", "fraud_West_Elm_LLC",
        "fraud_Pier_1_Imports", "fraud_HomeGoods_Inc", "fraud_Arhaus_Furniture",
        "fraud_Ashley_Furniture", "fraud_Sleep_Number_Corp", "fraud_Purple_Mattress",
        "fraud_Casper_Sleep_Inc", "fraud_Tempur_Pedic_Intl", "fraud_Kohler_Co",
        "fraud_Moen_Inc", "fraud_Andersen_Windows",
    ],
    "misc_pos": [
        "fraud_Dollar_General_Corp", "fraud_Dollar_Tree_Inc", "fraud_Five_Below_Inc",
        "fraud_Burlington_Stores", "fraud_Ross_Dress_Less", "fraud_Marshalls_LLC",
        "fraud_Tuesday_Morning_Corp", "fraud_Big_Lots_Inc", "fraud_Family_Dollar",
        "fraud_99_Cents_Only", "fraud_Harbor_Freight_Tools", "fraud_Ace_Hardware",
        "fraud_Office_Depot_Inc", "fraud_Staples_Inc", "fraud_Michaels_Stores",
        "fraud_Hobby_Lobby_Stores", "fraud_Joann_Stores_Inc", "fraud_Party_City_Corp",
    ],
    "misc_net": [
        "fraud_PayPal_Holdings", "fraud_Square_Inc", "fraud_Stripe_Payments",
        "fraud_Google_Pay_Svcs", "fraud_Apple_Pay_Svcs", "fraud_Zelle_Network",
        "fraud_Western_Union_Co", "fraud_MoneyGram_Intl", "fraud_Cash_App_Inc",
        "fraud_Venmo_LLC", "fraud_Coinbase_Global", "fraud_Robinhood_Markets",
        "fraud_Chime_Financial", "fraud_Green_Dot_Corp", "fraud_Netspend_Corp",
        "fraud_Amazon_Pay_Svcs", "fraud_Adyen_NV", "fraud_Checkout_com",
    ],
}
# Per-category cap/floor arrays aligned with ALL_CATEGORIES index order.
_CAT_CAPS = np.array([CATEGORY_MAX_AMT.get(c, 3000.0) for c in ALL_CATEGORIES], dtype=np.float64)
_CAT_MINS = np.array([CATEGORY_MIN_AMT.get(c, 1.0) for c in ALL_CATEGORIES], dtype=np.float64)

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

# Hardcoded fallback distributions
_HARDCODED_DEFAULTS: dict = {
    "fraud_rate": 0.005,
    "amount_legit_params": (0.8, 0.0, 45.0),
    "amount_fraud_params": (1.0, 0.0, 150.0),
    "hour_weights": [
        0.030, 0.020, 0.020, 0.020, 0.020, 0.030,
        0.040, 0.050, 0.060, 0.060, 0.060, 0.060,
        0.060, 0.060, 0.060, 0.060, 0.060, 0.060,
        0.060, 0.050, 0.050, 0.040, 0.040, 0.030,
    ],
    "fraud_hour_weights": [
        0.120, 0.100, 0.080, 0.070, 0.070, 0.080,
        0.060, 0.050, 0.050, 0.040, 0.040, 0.040,
        0.040, 0.040, 0.040, 0.040, 0.040, 0.040,
        0.030, 0.030, 0.030, 0.040, 0.040, 0.040,
    ],
    "category_weights": [1 / 14] * 14,
    "fraud_category_weights": [1 / 14] * 14,
    "lat_range": (24.0, 50.0),
    "long_range": (-125.0, -65.0),
    "merch_lat_range": (23.0, 51.0),
    "merch_long_range": (-126.0, -64.0),
    "city_pop_params": (1.2, 0.0, 8000.0),
    "unix_range": (1325376000, 1388534400),
    "zip_range": (1001, 99950),
}


# ---------------------------------------------------------------------------
# Distribution fitting (one-time CPU work using scipy)
# ---------------------------------------------------------------------------

def _build_identity_pools() -> tuple:
    """Build fixed cardholder pool. Merchant pools are now per-category (see CATEGORY_MERCHANTS)."""
    rng = np.random.default_rng()
    cc_num_pool = rng.integers(10**15, 10**16 - 1, NUM_USERS)
    return cc_num_pool, []  # merchant_pool unused; kept for signature compat


def _open_csv(seed_path: Path) -> pd.DataFrame:
    if seed_path.suffix == ".zip":
        with zipfile.ZipFile(seed_path) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {seed_path}")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(seed_path)


def load_seed_distributions(seed_csv_path: str) -> dict:
    """Fit distributions from seed data. Returns params dict (not raw data)."""
    path = Path(seed_csv_path)
    log.info("Loading seed distributions from: %s", path)
    try:
        df = _open_csv(path)

        if "Class" in df.columns and "Amount" in df.columns:
            df = df.rename(columns={"Class": "is_fraud", "Amount": "amt"})
            df["unix_time"] = df["Time"].astype(int) + 1378080000
            df["lat"] = np.nan
            df["long"] = np.nan
            df["merch_lat"] = np.nan
            df["merch_long"] = np.nan
            df["zip"] = 10000
            df["city_pop"] = 5000
            df["category"] = "misc_net"
        elif "is_fraud" not in df.columns:
            raise ValueError("Cannot detect schema")

        fraud = df[df["is_fraud"] == 1]
        legit = df[df["is_fraud"] == 0]
        fraud_rate = len(fraud) / len(df)
        log.info("Seed: %d rows, fraud_rate=%.4f", len(df), fraud_rate)

        hours_all = (df["unix_time"] % 86400) // 3600
        hours_fraud = (fraud["unix_time"] % 86400) // 3600
        hour_hist, _ = np.histogram(hours_all, bins=24, range=(0, 24), density=True)
        fraud_hour_hist, _ = np.histogram(hours_fraud, bins=24, range=(0, 24), density=True)
        hour_weights = (hour_hist / hour_hist.sum()).tolist()
        fraud_hour_weights = (fraud_hour_hist / fraud_hour_hist.sum()).tolist()

        cat_counts = df["category"].value_counts(normalize=True)
        fraud_cat_counts = fraud["category"].value_counts(normalize=True)
        category_weights = cat_counts.reindex(ALL_CATEGORIES).fillna(0.001).tolist()
        fraud_category_weights = fraud_cat_counts.reindex(ALL_CATEGORIES).fillna(0.0001).tolist()

        def safe_range(series, default):
            clean = series.dropna()
            return (float(clean.min()), float(clean.max())) if len(clean) > 0 else default

        lat_range = safe_range(df["lat"], _HARDCODED_DEFAULTS["lat_range"])
        long_range = safe_range(df["long"], _HARDCODED_DEFAULTS["long_range"])
        merch_lat_range = safe_range(df["merch_lat"], _HARDCODED_DEFAULTS["merch_lat_range"])
        merch_long_range = safe_range(df["merch_long"], _HARDCODED_DEFAULTS["merch_long_range"])

        city_pop_params = stats.lognorm.fit(df["city_pop"].clip(lower=1))
        unix_range = (int(df["unix_time"].min()), int(df["unix_time"].max()))
        zip_range = (int(df["zip"].min()), int(df["zip"].max()))

        return {
            "fraud_rate": fraud_rate,
            "hour_weights": hour_weights,
            "fraud_hour_weights": fraud_hour_weights,
            "category_weights": category_weights,
            "fraud_category_weights": fraud_category_weights,
            "lat_range": lat_range,
            "long_range": long_range,
            "merch_lat_range": merch_lat_range,
            "merch_long_range": merch_long_range,
            "city_pop_params": city_pop_params,
            "unix_range": unix_range,
            "zip_range": zip_range,
        }
    except Exception as exc:
        log.warning("Could not load seed data (%s) — using hardcoded defaults", exc)
        return _HARDCODED_DEFAULTS.copy()


# ---------------------------------------------------------------------------
# GPU generation
# ---------------------------------------------------------------------------

def _make_cumprobs(weights, cp):
    """Convert probability weights to cumulative probabilities on GPU."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    c = np.cumsum(w)
    c[-1] = 1.0  # ensure exact 1.0 to avoid searchsorted overflow
    return cp.asarray(c)


def _weighted_choice(cumprobs, rng, n, cp):
    """GPU-accelerated weighted random sampling via CDF inversion."""
    u = cp.asarray(rng.random_sample(n), dtype=cp.float64)
    idx = cp.searchsorted(cumprobs, u)
    return cp.minimum(idx, len(cumprobs) - 1).astype(cp.int32)


def _build_gpu_pools(dist, cc_num_pool, merchant_pool, cp):
    """Pre-build GPU-resident arrays and string pools for generation."""
    faker = Faker("en_US")
    _rng = np.random.default_rng()

    # String pools as numpy arrays for fast fancy indexing.
    first_pool = np.array([faker.first_name() for _ in range(500)])
    last_pool = np.array([faker.last_name() for _ in range(500)])
    street_pool = np.array([faker.street_address()[:50] for _ in range(500)])
    city_pool = np.array([faker.city()[:30] for _ in range(500)])
    job_pool = np.array([faker.job()[:40] for _ in range(500)])
    trans_pool = np.array([f"{_rng.integers(10**15, 10**16):016x}" for _ in range(100_000)])
    state_pool = np.array(US_STATES)
    cat_pool = np.array(ALL_CATEGORIES)

    # Per-category merchant pools: one numpy array of strings per category,
    # aligned with ALL_CATEGORIES index order for O(1) lookup by cat_idx.
    cat_merch_pools = [
        np.array(CATEGORY_MERCHANTS[cat]) for cat in ALL_CATEGORIES
    ]

    # GPU-resident cumulative probability arrays for weighted sampling.
    gpu_dist = {
        "fraud_rate": dist.get("fraud_rate", 0.005),
        "hour_cumprobs": _make_cumprobs(dist["hour_weights"], cp),
        "fraud_hour_cumprobs": _make_cumprobs(dist["fraud_hour_weights"], cp),
        "cat_cumprobs": _make_cumprobs(dist["category_weights"], cp),
        "fraud_cat_cumprobs": _make_cumprobs(dist["fraud_category_weights"], cp),
        "lat_range": dist["lat_range"],
        "long_range": dist["long_range"],
        "merch_lat_range": dist["merch_lat_range"],
        "merch_long_range": dist["merch_long_range"],
        "city_pop_sigma": dist["city_pop_params"][0],  # lognorm shape param
        "city_pop_scale": dist["city_pop_params"][2],   # lognorm scale
        "city_pop_loc": dist["city_pop_params"][1],     # lognorm loc
        "unix_range": dist["unix_range"],
        "zip_range": dist["zip_range"],
        "cat_caps_gpu": cp.asarray(_CAT_CAPS),
        "cat_mins_gpu": cp.asarray(_CAT_MINS),
        "high_fraud_mask_set": HIGH_FRAUD_INDICES,
    }

    pools = {
        "cc_num": cp.asarray(cc_num_pool),  # int64 on GPU
        "cat_merch": cat_merch_pools,        # list[np.ndarray], one per category
        "first": first_pool,
        "last": last_pool,
        "street": street_pool,
        "city": city_pool,
        "job": job_pool,
        "trans": trans_pool,
        "state": state_pool,
        "cat": cat_pool,
    }

    return gpu_dist, pools


def generate_chunk_gpu(chunk_id, n_rows, seed_offset, gpu_dist, pools, cudf, cp):
    """Generate one chunk of synthetic transactions entirely on GPU.

    Numeric columns: cupy random generation (GPU kernels).
    String columns: GPU index generation → CPU pool lookup → cudf transfer.
    Returns: (cudf.DataFrame, actual_fraud_rate)
    """
    seed = int((seed_offset + chunk_id * 997) & 0xFFFFFFFF)
    rng = cp.random.RandomState(seed)
    n = n_rows
    d = gpu_dist

    # ── Fraud flags ──────────────────────────────────────────────────────
    is_fraud = rng.random_sample(n) < d["fraud_rate"]

    # ── Categories (weighted choice, fraud-biased) ───────────────────────
    fraud_cat_idx = _weighted_choice(d["fraud_cat_cumprobs"], rng, n, cp)
    legit_cat_idx = _weighted_choice(d["cat_cumprobs"], rng, n, cp)
    cat_idx = cp.where(is_fraud, fraud_cat_idx, legit_cat_idx)

    # Boost fraud for high-fraud categories (3×)
    high_fraud_mask = cp.zeros(n, dtype=cp.bool_)
    for hi in d["high_fraud_mask_set"]:
        high_fraud_mask |= (cat_idx == hi)
    extra_fraud = (~is_fraud) & high_fraud_mask & (rng.random_sample(n) < d["fraud_rate"] * 2)
    is_fraud = is_fraud | extra_fraud

    # ── Hours (fraud skews to night) ─────────────────────────────────────
    fraud_hours = _weighted_choice(d["fraud_hour_cumprobs"], rng, n, cp)
    legit_hours = _weighted_choice(d["hour_cumprobs"], rng, n, cp)
    hours = cp.where(is_fraud, cp.minimum(fraud_hours, 23), cp.minimum(legit_hours, 23))

    # ── Unix timestamps ──────────────────────────────────────────────────
    lo, hi = d["unix_range"]
    base_unix = rng.randint(lo, hi, n).astype(cp.int64)
    unix_time = (base_unix - (base_unix % 86400)) + hours.astype(cp.int64) * 3600 + rng.randint(0, 3600, n).astype(cp.int64)

    # ── Amounts (category-aware caps and floors) ──────────────────────────
    caps = d["cat_caps_gpu"][cat_idx]
    mins = d["cat_mins_gpu"][cat_idx]
    raw_legit = rng.lognormal(0.0, 0.8, n) / 5.0
    raw_fraud = rng.lognormal(0.0, 1.0, n) / 4.0
    raw_legit = cp.clip(raw_legit, 0.0, 1.0)
    raw_fraud = cp.clip(raw_fraud, 0.0, 1.0)
    amt = cp.where(is_fraud, raw_fraud * caps, raw_legit * caps)
    amt = cp.clip(amt, mins, caps)
    amt = cp.around(amt, 2)

    # ── Geographic coordinates ───────────────────────────────────────────
    lat = rng.uniform(d["lat_range"][0], d["lat_range"][1], n)
    long_ = rng.uniform(d["long_range"][0], d["long_range"][1], n)
    merch_lat_normal = rng.uniform(d["merch_lat_range"][0], d["merch_lat_range"][1], n)
    merch_lon_normal = rng.uniform(d["merch_long_range"][0], d["merch_long_range"][1], n)
    sign1 = (rng.randint(0, 2, n) * 2 - 1).astype(cp.float64)
    sign2 = (rng.randint(0, 2, n) * 2 - 1).astype(cp.float64)
    far_lat = lat + rng.uniform(4.5, 12.0, n) * sign1
    far_lon = long_ + rng.uniform(5.0, 18.0, n) * sign2
    merch_lat = cp.where(is_fraud, cp.clip(far_lat, 23.0, 51.0), merch_lat_normal)
    merch_long = cp.where(is_fraud, cp.clip(far_lon, -126.0, -64.0), merch_lon_normal)

    # ── City population (lognormal on GPU) ───────────────────────────────
    city_pop = rng.lognormal(0.0, d["city_pop_sigma"], n) * d["city_pop_scale"] + d["city_pop_loc"]
    city_pop = cp.clip(city_pop, 100, 5_000_000).astype(cp.int32)

    # ── Zip codes & merch_zipcode ────────────────────────────────────────
    zip_lo, zip_hi = d["zip_range"]
    zip_codes = rng.randint(zip_lo, zip_hi, n).astype(cp.int32)
    merch_zipcode = cp.where(high_fraud_mask, cp.nan, zip_codes.astype(cp.float64))

    # ── CC nums from shared pool (GPU-resident) ──────────────────────────
    cc_idx = rng.randint(0, len(pools["cc_num"]), n)
    cc_nums = pools["cc_num"][cc_idx]

    # ── String columns: GPU index generation → CPU pool lookup ───────────
    cat_idx_cpu = cp.asnumpy(cat_idx)
    categories = pools["cat"][cat_idx_cpu]

    state_idx = cp.asnumpy(rng.randint(0, len(US_STATES), n))
    states = pools["state"][state_idx]

    gender_cpu = cp.asnumpy(rng.random_sample(n) < 0.55)
    genders = np.where(gender_cpu, "F", "M")

    # Category-aware merchant selection: generate indices directly in each
    # pool's range (no modulo) to avoid bias clustering on specific merchants.
    merchants = np.empty(n, dtype=object)
    for ci, cat_pool_arr in enumerate(pools["cat_merch"]):
        mask = cat_idx_cpu == ci
        count = int(mask.sum())
        if count > 0:
            within = cp.asnumpy(rng.randint(0, len(cat_pool_arr), count))
            merchants[mask] = cat_pool_arr[within]

    fi = cp.asnumpy(rng.randint(0, len(pools["first"]), n))
    firsts = pools["first"][fi]

    li = cp.asnumpy(rng.randint(0, len(pools["last"]), n))
    lasts = pools["last"][li]

    si = cp.asnumpy(rng.randint(0, len(pools["street"]), n))
    streets = pools["street"][si]

    ci = cp.asnumpy(rng.randint(0, len(pools["city"]), n))
    cities = pools["city"][ci]

    ji = cp.asnumpy(rng.randint(0, len(pools["job"]), n))
    jobs = pools["job"][ji]

    ti = cp.asnumpy(rng.randint(0, len(pools["trans"]), n))
    trans_nums = pools["trans"][ti]

    # ── Date strings (GPU→CPU→pandas datetime formatting) ────────────────
    unix_cpu = cp.asnumpy(unix_time)
    trans_dates = pd.to_datetime(unix_cpu, unit="s").strftime("%Y-%m-%d %H:%M:%S").values

    dob_lo = max(int(d["unix_range"][0]) - 80 * 365 * 86400, 0)
    dob_hi = int(d["unix_range"][0]) - 18 * 365 * 86400
    dob_unix = cp.asnumpy(rng.randint(dob_lo, dob_hi, n).astype(cp.int64))
    dob = pd.to_datetime(dob_unix, unit="s").strftime("%Y-%m-%d").values

    # ── Build cudf DataFrame (cupy arrays stay on GPU, numpy arrays transferred) ──
    gdf = cudf.DataFrame({
        "trans_date_trans_time": trans_dates,
        "cc_num": cc_nums,
        "merchant": merchants,
        "category": categories,
        "amt": amt,
        "first": firsts,
        "last": lasts,
        "gender": genders,
        "street": streets,
        "city": cities,
        "state": states,
        "zip": zip_codes,
        "lat": lat,
        "long": long_,
        "city_pop": city_pop,
        "job": jobs,
        "dob": dob,
        "trans_num": trans_nums,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "is_fraud": is_fraud.astype(cp.int8),
        "merch_zipcode": merch_zipcode,
        "chunk_ts": cp.full(n, time.time(), dtype=cp.float64),
    })

    fraud_rate_actual = float(is_fraud.mean())
    return gdf, fraud_rate_actual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_disk_space(path: Path) -> tuple:
    """Returns (usage_fraction, should_stop)."""
    try:
        import psutil
        check_path = path if path.exists() else path.parent
        usage = psutil.disk_usage(str(check_path))
        pct = usage.percent / 100.0
        test_file = check_path / ".disk_check_probe"
        try:
            test_file.write_bytes(b"x")
            test_file.unlink(missing_ok=True)
            return pct, False
        except OSError:
            return pct, True
    except Exception:
        return 0.0, False


def emit_telemetry(total_rows, total_bytes, files_written, fraud_rate,
                   start_time, rows_since_last=0, elapsed_since_last=1.0):
    elapsed = max(time.time() - start_time, 0.001)
    throughput_mbps = (total_bytes / 1e6) / elapsed
    rows_per_sec = int(rows_since_last / max(elapsed_since_last, 0.001))
    sys.stdout.write(
        f"[TELEMETRY] stage=gather rows_generated={total_rows} "
        f"rows_per_sec={rows_per_sec} "
        f"throughput_mbps={throughput_mbps:.1f} files_written={files_written} "
        f"gpu_used=1 fraud_rate={fraud_rate:.4f}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _SHUTDOWN

    # GPU imports deferred — same pattern as data-prep to avoid CUDA in importers.
    import cudf
    import cupy as cp

    log.info("GPU gather: cudf %s, cupy %s, CUDA device %d",
             cudf.__version__, cp.__version__, cp.cuda.Device().id)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        OUTPUT_PATH.chmod(0o777)
    except PermissionError:
        pass  # RAPIDS container runs as non-root; dir already writable from gather pod

    # Load seed distributions (one-time CPU work with scipy).
    if KAGGLE_SEED_PATH and Path(KAGGLE_SEED_PATH).exists():
        dist = load_seed_distributions(KAGGLE_SEED_PATH)
    else:
        log.warning("KAGGLE_SEED_PATH not set — using hardcoded defaults")
        dist = _HARDCODED_DEFAULTS.copy()

    # Build identity pools and GPU-resident distribution arrays.
    cc_num_pool, merchant_pool = _build_identity_pools()
    log.info("Identity pools: %d users, %d merchant categories", NUM_USERS, len(ALL_CATEGORIES))
    gpu_dist, pools = _build_gpu_pools(dist, cc_num_pool, merchant_pool, cp)

    chunk_size = CHUNK_SIZE
    target_rows_per_sec = TARGET_ROWS_PER_SEC
    target_chunk_time = (chunk_size / target_rows_per_sec) if target_rows_per_sec > 0 else 0.0

    log.info("Starting GPU data-gather: mode=%s chunk_size=%d target_rows_per_sec=%d",
             RUN_MODE, chunk_size, target_rows_per_sec)

    total_rows = 0
    total_bytes = 0.0
    files_written = 0
    chunks_since_disk_check = 0
    start_time = time.time()
    last_telemetry = start_time
    rows_at_last_telemetry = 0
    actual_fraud_rate = dist.get("fraud_rate", FRAUD_RATE)
    seed_offset = int(time.time()) & 0xFFFFFF
    chunk_id = 0

    def _process_one_chunk():
        nonlocal chunk_id, total_rows, total_bytes, files_written, chunks_since_disk_check
        nonlocal last_telemetry, rows_at_last_telemetry, actual_fraud_rate

        Path("/tmp/.healthy").touch()
        chunk_start = time.time()

        # Disk check every 10 chunks.
        chunks_since_disk_check += 1
        if chunks_since_disk_check >= 10:
            chunks_since_disk_check = 0
            usage_pct, should_stop = check_disk_space(OUTPUT_PATH)
            if should_stop:
                sys.stdout.write(
                    f"[TELEMETRY] stage=gather status=PAUSED reason=disk_full "
                    f"disk_usage_pct={usage_pct * 100:.1f}\n"
                )
                sys.stdout.flush()
                log.warning("Disk full — pausing 60s")
                time.sleep(60)
                return False
            if usage_pct > 0.8:
                log.warning("Disk usage %.0f%% > 80%%", usage_pct * 100)

        # Generate chunk on GPU.
        gdf, actual_fraud_rate = generate_chunk_gpu(
            chunk_id, chunk_size, seed_offset, gpu_dist, pools, cudf, cp
        )
        n = len(gdf)

        # Write directly from GPU via cudf parquet writer.
        fname = f"{HOSTNAME}_raw_chunk_{files_written:06d}.parquet"
        out_file = OUTPUT_PATH / fname
        tmp_file = out_file.with_suffix(".parquet.tmp")
        gdf.to_parquet(str(tmp_file))
        del gdf
        Path(tmp_file).rename(out_file)

        total_rows += n
        try:
            total_bytes += out_file.stat().st_size
        except FileNotFoundError:
            pass
        files_written += 1
        chunk_id += 1

        # Telemetry every 1s.
        now = time.time()
        if now - last_telemetry >= 1.0:
            emit_telemetry(total_rows, total_bytes, files_written,
                           actual_fraud_rate, start_time,
                           rows_since_last=total_rows - rows_at_last_telemetry,
                           elapsed_since_last=now - last_telemetry)
            rows_at_last_telemetry = total_rows
            last_telemetry = now

        # Rate governor.
        if target_chunk_time > 0:
            jitter = np.random.triangular(0.55, 0.92, 1.15)
            sleep_for = target_chunk_time * jitter - (time.time() - chunk_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
        return True

    if RUN_MODE == "once":
        n_chunks = (TARGET_ROWS + chunk_size - 1) // chunk_size
        for _ in range(n_chunks):
            if _SHUTDOWN:
                break
            _process_one_chunk()
    else:
        # Continuous mode.
        while not _SHUTDOWN:
            if not _process_one_chunk():
                continue  # disk-full pause, retry

    emit_telemetry(total_rows, total_bytes, files_written, actual_fraud_rate, start_time)
    log.info("GPU data-gather complete: %d rows, %d files, %.1fs",
             total_rows, files_written, time.time() - start_time)


if __name__ == "__main__":
    main()
