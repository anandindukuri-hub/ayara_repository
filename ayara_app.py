import os
import io
import re
import glob
import lzma
import json
import time
import threading
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False



def _resolve_base_dir():
    """Locate the 'Resources' folder that holds the images, the dataset, and the
    'Time Series Models' subfolder.

    Resolution order (first one that actually exists wins):
      1. AYARA_RESOURCES_DIR environment variable  (manual override, optional)
      2. ~/Documents/Resources                     (the standard location)
      3. ~/OneDrive/Documents/Resources            (Windows OneDrive-redirected Documents)
      4. ./Resources                               (legacy relative path)
    If none exist yet, the standard ~/Documents/Resources path is returned so any
    "file not found" messages point you to the expected location.
    """
    home = os.path.expanduser("~")
    candidates = []
    env_override = os.environ.get("AYARA_RESOURCES_DIR")
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(home, "Documents", "Resources"))
    candidates.append(os.path.join(home, "OneDrive", "Documents", "Resources"))
    candidates.append("Resources")
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.join(home, "Documents", "Resources")


BASE_DIR = _resolve_base_dir()
LOGO_PATH = os.path.join(BASE_DIR, "Ayara.png")
INFLATION_IMG_PATH = os.path.join(BASE_DIR, "Inflation.jpg")
DATA_PATH = os.path.join(BASE_DIR, "Revised Final Dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Time Series Models")


GEMINI_MODEL = "gemini-2.5-flash"


MAX_REQUESTS_PER_MINUTE = 8     
MAX_REQUESTS_PER_DAY = 200      


MAX_UPLOAD_MB = 7               
MAX_FILES_PER_RUN = 8           


USAGE_FILE = os.path.join(os.path.expanduser("~"), ".ayara_gemini_usage.json")

PRODUCT_LIST = ["Fruits", "Vegetables", "Spices", "Fuel and light", "Health"]
SECTORS = ["Urban", "Rural", "Rural+Urban"]

MODEL_PATHS = {
    ("Vegetables", "Urban"): os.path.join(MODELS_DIR, "vegetables-urban.pkl"),
    ("Vegetables", "Rural"): os.path.join(MODELS_DIR, "vegetables-rural.pkl"),
    ("Vegetables", "Rural+Urban"): os.path.join(MODELS_DIR, "Vegetables-Total.pkl"),
    ("Fuel and light", "Urban"): os.path.join(MODELS_DIR, "Fuel and light-urban.pkl"),
    ("Fuel and light", "Rural"): os.path.join(MODELS_DIR, "Fuel and light-rural.pkl"),
    ("Fuel and light", "Rural+Urban"): os.path.join(MODELS_DIR, "Fuel and light-total.pkl"),
    ("Health", "Urban"): os.path.join(MODELS_DIR, "Health-urban.pkl"),
    ("Health", "Rural"): os.path.join(MODELS_DIR, "Health-rural.pkl"),
    ("Health", "Rural+Urban"): os.path.join(MODELS_DIR, "Health-Total.pkl"),
    ("Fruits", "Urban"): os.path.join(MODELS_DIR, "Fruits-urban.pkl"),
    ("Fruits", "Rural"): os.path.join(MODELS_DIR, "Fruits-Rural.pkl"),
    ("Fruits", "Rural+Urban"): os.path.join(MODELS_DIR, "Fruits-Total.pkl"),
    ("Spices", "Urban"): os.path.join(MODELS_DIR, "Spices-Urban.pkl"),
    ("Spices", "Rural"): os.path.join(MODELS_DIR, "spices-rural.pkl"),
    ("Spices", "Rural+Urban"): os.path.join(MODELS_DIR, "spices-total.pkl"),
}


def _model_name_variants(pkl_path):
    """Filename variants to try, bridging space/underscore naming differences
    (e.g. 'Fuel and light-urban.pkl' vs 'Fuel_and_light-urban.pkl')."""
    folder, base = os.path.split(pkl_path)
    names = [base]
    if " " in base:
        names.append(base.replace(" ", "_"))
    if "_" in base:
        names.append(base.replace("_", " "))
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(os.path.join(folder, n))
    return out


def load_model(pkl_path):
    """Load a forecasting model, transparently handling three on-disk forms that
    all deserialize to the IDENTICAL object (so forecasts/graphs never change):

        1. <name>.pkl                    uncompressed  -> loaded directly
        2. <name>.pkl.xz                 lzma-compressed -> decompressed in memory
        3. <name>.pkl.xz.part001, 002…   compressed + split into <25 MB chunks
                                         -> chunks joined, then decompressed

    Also bridges space/underscore filename differences. The decompressed bytes
    are exactly the original pickle bytes, so joblib rebuilds the same model.
    Raises FileNotFoundError if no matching file is found.
    """
    for cand in _model_name_variants(pkl_path):
        # 1) plain uncompressed pickle (kept for backward compatibility)
        if os.path.exists(cand):
            return joblib.load(cand)
        # 2) single lzma-compressed file
        xz = cand + ".xz"
        if os.path.exists(xz):
            with lzma.open(xz, "rb") as fh:
                return joblib.load(io.BytesIO(fh.read()))
        # 3) split, lzma-compressed chunks (join in filename order, then decompress)
        parts = sorted(glob.glob(cand + ".xz.part*"))
        if parts:
            blob = b"".join(open(p, "rb").read() for p in parts)
            return joblib.load(io.BytesIO(lzma.decompress(blob)))
    raise FileNotFoundError(
        "No model file found for '%s'. Looked for the .pkl, a .pkl.xz, or "
        ".pkl.xz.partNNN chunks (including space/underscore name variants) in %s."
        % (os.path.basename(pkl_path), os.path.dirname(pkl_path) or ".")
    )


def update_to_data_end(model, observed):
    """The saved models were fit on data through July 2024. Before a forecast, the newer months in the
    dataset are added to the model with its fitted settings unchanged (no refitting). Forecast steps then
    start the month after the data ends, which is where the app counts them from."""
    newer = observed[observed.index > model.data.row_labels[-1]]
    if newer.empty:
        return model
    return model.append(newer.asfreq("MS"), refit=False)


st.set_page_config(layout="wide", page_title="Ayara")



def safe_image(path, caption=None):
    
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f"Image not found at: {path}\n\nUpdate the path in the CONFIG block to display it.")


@st.cache_data(show_spinner=False)
def load_data(path):
    
    data = pd.read_csv(path)
    data["Month_Year"] = pd.to_datetime(data["Month_Year"])
    return data


def get_data():
    
    try:
        return load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find the dataset at: {DATA_PATH}\n\nUpdate DATA_PATH in the CONFIG block.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
        st.stop()


def get_api_key():
    
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def key_looks_valid(key):
    
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    if "InsertAPIKEY" in k or k.lower() == "insertapikeyhere":
        return False
    return k.startswith("AIza") and len(k) >= 30


def ai_ready():
    
    return GEMINI_AVAILABLE and key_looks_valid(get_api_key())



@st.cache_resource
def _rate_state():
    
    return {"minute_times": [], "lock": threading.Lock()}


def _pacific_today():
    
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        
        return (datetime.utcnow() - timedelta(hours=8)).strftime("%Y-%m-%d")


def _read_day_usage():
    
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        if data.get("date") == _pacific_today():
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _write_day_usage(count):
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump({"date": _pacific_today(), "count": int(count)}, f)
    except Exception:
        pass  

def reserve_quota():
 
    state = _rate_state()
    now = time.time()
    with state["lock"]:
        
        recent = [t for t in state["minute_times"] if now - t < 60]
        if len(recent) >= MAX_REQUESTS_PER_MINUTE:
            wait = int(60 - (now - min(recent))) + 1
            state["minute_times"] = recent
            return False, (
                f"Per-minute safety limit reached ({MAX_REQUESTS_PER_MINUTE}/min). "
                f"Please wait about {wait} seconds and try again."
            )
        
        day_count = _read_day_usage()
        if day_count >= MAX_REQUESTS_PER_DAY:
            return False, (
                f"Daily safety limit reached ({MAX_REQUESTS_PER_DAY} requests today). "
                
            )

        recent.append(now)
        state["minute_times"] = recent
        _write_day_usage(day_count + 1)
        return True, ""


def quota_status():
   
    return _read_day_usage(), MAX_REQUESTS_PER_DAY



def _gemini_generate(contents, response_json=False, max_retries=3):
   
    if not GEMINI_AVAILABLE:
        return "__ERROR__The google-genai package is not installed. Run: pip install google-genai"
    key = get_api_key()
    if not key_looks_valid(key):
        return "__ERROR__No valid Gemini API key found. Add GEMINI_API_KEY to .streamlit/secrets.toml"

    allowed, reason = reserve_quota()
    if not allowed:
        return f"__ERROR__{reason}"

    config = None
    if response_json:
        try:
            config = genai_types.GenerateContentConfig(response_mime_type="application/json")
        except Exception:
            config = None

    last_err = ""
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=key)
            if config is not None:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents, config=config
                )
            else:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents
                )
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            return "__ERROR__The model returned an empty response. Please try again."
        except Exception as e:
            
            msg = str(e)
            transient = any(code in msg for code in ("429", "503", "500", "deadline", "timeout"))
            last_err = "rate limit or temporary server error" if transient else "request failed"
            if transient and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  
                continue
            break
    return f"__ERROR__Gemini {last_err}. Please try again in a moment."


@st.cache_data(show_spinner=False)
def ai_translate_simplify(text, language, simplify):
    
    if not GEMINI_AVAILABLE or not key_looks_valid(get_api_key()):
        return None
    if simplify:
        instruction = (
            "Rewrite the inflation analysis below so it is noticeably shorter, clearer, and easy "
            "for a general reader to understand. Keep every important fact, number, and recommendation, "
            "remove repetition and jargon, and use plain, friendly language. "
            f"Write the final result in {language}."
        )
    else:
        instruction = (
            f"Translate the inflation analysis below into {language}. "
            "Preserve all facts, numbers, structure, and meaning. Do not add any commentary."
        )
    prompt = (
        f"{instruction}\n\n"
        "Return ONLY the rewritten/translated text, with no preamble or explanation.\n\n"
        f"---\n{text}"
    )
    return _gemini_generate(prompt)


@st.cache_data(show_spinner=False)
def ocr_receipt(file_bytes, mime_type):
    
    if not GEMINI_AVAILABLE or not key_looks_valid(get_api_key()):
        return {"__error__": "Gemini is not configured. Add GEMINI_API_KEY to enable receipt reading."}

    prompt = (
        "You are a receipt parser. Read this receipt and return ONLY JSON (no markdown, "
        "no explanation) with exactly these fields:\n"
        "{\n"
        '  "merchant_name": string or null,\n'
        '  "transaction_date": "YYYY-MM-DD" or null,\n'
        '  "currency": string or null,\n'
        '  "total": number or null,\n'
        '  "tax": number or null,\n'
        '  "items": [ { "description": string, "quantity": number, '
        '"unit_price": number, "line_total": number } ]\n'
        "}\n"
        "Normalize dates to YYYY-MM-DD. If a value is missing, use null. "
        "Return only the JSON object."
    )

    try:
        part = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    except Exception:
        return {"__error__": "Could not prepare the file for reading. Try a different image."}

    raw = _gemini_generate([prompt, part], response_json=True)
    if isinstance(raw, str) and raw.startswith("__ERROR__"):
        return {"__error__": raw[len("__ERROR__"):]}

    
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return {"__error__": "The model did not return receipt fields. Try a clearer photo."}
        return data
    except Exception:
        return {"__error__": "Could not understand the receipt. Try a clearer, well-lit photo."}



# ============================================================================
# CPI INTERPRETATION  ("Know more" / interpret-this-graph)
# ============================================================================
# These helpers turn the numbers behind a forecast into a plain-language
# explanation of what the CPI value means, plus practical, trend-aware advice.
# The core explanation is computed locally and ALWAYS works (no API key and no
# internet required). When a Gemini key is present, an optional button adds
# AI-tailored advice. Only the derived NUMBERS are ever sent to the AI -- never
# receipts, names, or other personal data -- so the feature is privacy-safe.

# Short, practical, category-specific tips used in the interpretation panel.
PRODUCT_FORECAST_TIPS = {
    "Fruits": (
        "Favour in-season fruits, buy durable varieties such as apples and citrus in bulk "
        "when they are cheap, and use cold storage where you can. Keep an eye on weather news "
        "for major fruit-growing regions, since that often moves prices first."
    ),
    "Vegetables": (
        "Stock staples that keep well such as onions and potatoes, buy at local mandis or "
        "farmers' markets where prices are usually lower, and switch to whichever seasonal "
        "vegetable is cheapest when a particular item spikes."
    ),
    "Spices": (
        "Buy your annual stock soon after harvest (for example cumin and coriander between "
        "March and May), store everything airtight, and substitute or simply use a little less "
        "of any spice that becomes expensive."
    ),
    "Fuel and light": (
        "Combine errands into single trips, switch to energy-efficient appliances and LED "
        "lighting, consider public transport or carpooling, and shift heavy usage away from "
        "peak-demand months where possible."
    ),
    "Health": (
        "Use government schemes such as Ayushman Bharat, buy generic medicines from Jan Aushadhi "
        "stores, keep essential medicines at home ahead of high-illness seasons, and maintain "
        "health insurance to cushion sudden large bills."
    ),
}


def _fmt_month(ts):
    """Format an index value as 'Month Year' when possible, else fall back to str."""
    try:
        return pd.Timestamp(ts).strftime("%B %Y")
    except Exception:
        return str(ts)


def compute_cpi_interpretation_stats(observed, forecast_mean, forecast_ci, product, sector):
    """Derive the key numbers used to explain a CPI forecast.

    All values are plain Python floats/strings so they can be formatted in text
    or sent to the AI helper safely. Returns a dict; any missing piece is None.
    """
    stats = {"product": product, "sector": sector, "base_index": 100.0}

    try:
        obs = observed.dropna()
    except Exception:
        obs = observed

    if obs is not None and len(obs):
        stats["first_obs_value"] = float(obs.iloc[0])
        stats["first_obs_date"] = _fmt_month(obs.index[0])
        stats["last_obs_value"] = float(obs.iloc[-1])
        stats["last_obs_date"] = _fmt_month(obs.index[-1])
    else:
        stats["first_obs_value"] = stats["first_obs_date"] = None
        stats["last_obs_value"] = stats["last_obs_date"] = None

    _fc_end = None
    if forecast_mean is not None and len(forecast_mean):
        stats["forecast_value"] = float(forecast_mean.iloc[-1])
        stats["forecast_date"] = _fmt_month(forecast_mean.index[-1])
        _fc_end = forecast_mean.index[-1]
    else:
        stats["forecast_value"] = stats["forecast_date"] = None

    stats["forecast_low"] = stats["forecast_high"] = None
    try:
        if forecast_ci is not None and len(forecast_ci):
            stats["forecast_low"] = float(forecast_ci.iloc[-1, 0])
            stats["forecast_high"] = float(forecast_ci.iloc[-1, 1])
    except Exception:
        pass

    fv = stats["forecast_value"]

    # "What does a CPI of X translate to?" -- always relative to the base (=100).
    if fv is not None:
        stats["pct_above_base"] = fv - 100.0
        stats["purchasing_power_of_100"] = (100.0 * (100.0 / fv)) if fv else None
    else:
        stats["pct_above_base"] = stats["purchasing_power_of_100"] = None

    # Change from the latest real data point to the forecast month.
    lv = stats["last_obs_value"]
    stats["abs_change"] = stats["pct_change"] = None
    stats["horizon_months"] = stats["annualized_pct"] = None
    if lv is not None and fv is not None:
        stats["abs_change"] = fv - lv
        stats["pct_change"] = ((fv - lv) / lv * 100.0) if lv else None
        try:
            last_idx = obs.index[-1]
            months = (_fc_end.year - last_idx.year) * 12 + (_fc_end.month - last_idx.month)
            stats["horizon_months"] = int(months)
            if months and months > 0 and lv > 0:
                stats["annualized_pct"] = ((fv / lv) ** (12.0 / months) - 1.0) * 100.0
        except Exception:
            pass

    # Recent 12-month momentum on the observed data.
    stats["yoy_pct"] = None
    try:
        if obs is not None and len(obs) >= 13:
            prev = float(obs.iloc[-13])
            if prev:
                stats["yoy_pct"] = (float(obs.iloc[-1]) - prev) / prev * 100.0
    except Exception:
        pass

    # Volatility: std-dev of monthly % changes over the last two years.
    stats["volatility_pct"] = None
    try:
        if obs is not None and len(obs) >= 6:
            pct = obs.pct_change().dropna() * 100.0
            stats["volatility_pct"] = float(pct.tail(24).std())
    except Exception:
        pass

    # Direction / strength classification, used to choose the right advice.
    direction = "stable"
    pc = stats.get("pct_change")
    if pc is not None:
        if pc >= 1.0:
            direction = "rising"
        elif pc <= -1.0:
            direction = "falling"
    stats["direction"] = direction

    ann = stats.get("annualized_pct")
    if ann is None:
        strength = "mild"
    elif abs(ann) >= 8.0:
        strength = "steep"
    elif abs(ann) >= 3.0:
        strength = "moderate"
    else:
        strength = "mild"
    stats["strength"] = strength

    vol = stats.get("volatility_pct")
    stats["is_volatile"] = bool(vol is not None and vol >= 3.0)

    return stats


def _trend_sentence(stats):
    direction = stats.get("direction", "stable")
    strength = stats.get("strength", "mild")
    if direction == "rising" and strength == "steep":
        return ("Because the forecast points clearly upward, it is worth acting earlier rather "
                "than later: stock up or lock in prices where that is practical, and build the "
                "expected increase into your budget now.")
    if direction == "rising":
        return ("The forecast drifts gently upward, so light planning is enough: keep an eye on "
                "prices and adjust your budget gradually over the coming months.")
    if direction == "falling":
        return ("The forecast eases downward, so you may benefit from waiting on large or "
                "easily-stored purchases until prices settle a little lower.")
    return ("The forecast is fairly flat, so no urgent action is needed: keep your usual buying "
            "habits and revisit this if conditions change.")


def build_cpi_interpretation_text(stats):
    """Plain-language interpretation of a CPI forecast. Always available, no AI needed."""
    product = stats.get("product", "this category")
    sector = stats.get("sector", "")
    sec = sector.lower() if sector else ""
    prod = product.lower()

    lines = [f"### What this CPI means: {sector} {product}".strip()]

    fv = stats.get("forecast_value")
    fdate = stats.get("forecast_date")
    if fv is not None:
        pct_base = stats.get("pct_above_base")
        pp = stats.get("purchasing_power_of_100")
        base_phrase = ""
        if pct_base is not None:
            if pct_base >= 0:
                base_phrase = (f"about **{pct_base:.0f}% higher** than in the base period "
                               "(when the index equals 100)")
            else:
                base_phrase = (f"about **{abs(pct_base):.0f}% lower** than in the base period "
                               "(when the index equals 100)")
        msg = (f"A CPI of **{fv:.1f}**"
               + (f" for **{fdate}**" if fdate else "")
               + (f" means prices for {sec} {prod} are {base_phrase}." if base_phrase else "."))
        lines.append(msg)
        if pp is not None:
            lines.append(
                f"In everyday terms: what cost **₹100** in the base period would cost about "
                f"**₹{fv:.0f}** at this level, and **₹100 today** stretches only about as far as "
                f"**₹{pp:.0f}** did back in the base period."
            )

    lv = stats.get("last_obs_value")
    ldate = stats.get("last_obs_date")
    if lv is not None and fv is not None:
        abs_ch = stats.get("abs_change") or 0.0
        pct_ch = stats.get("pct_change")
        ann = stats.get("annualized_pct")
        horizon = stats.get("horizon_months")
        move = "rise" if abs_ch > 0 else ("fall" if abs_ch < 0 else "stay about level")
        seg = (f"Compared with the most recent actual reading of **{lv:.1f}** in **{ldate}**, "
               f"the model expects prices to {move} to **{fv:.1f}**")
        if pct_ch is not None:
            seg += f", a change of **{pct_ch:+.1f}%**"
        if horizon:
            seg += f" over **{horizon} month{'s' if horizon != 1 else ''}**"
        seg += "."
        lines.append(seg)
        if ann is not None:
            lines.append(f"That works out to roughly **{ann:+.1f}% per year** at the forecast pace.")

    lo = stats.get("forecast_low")
    hi = stats.get("forecast_high")
    if lo is not None and hi is not None:
        lines.append(
            f"The shaded band on the chart is the model's confidence range: the value for "
            f"{fdate or 'that month'} is most likely between **{lo:.1f}** and **{hi:.1f}**. "
            "A wider band means more uncertainty."
        )

    yoy = stats.get("yoy_pct")
    if yoy is not None:
        lines.append(
            f"Recent momentum: over the last 12 months of actual data, {sec} {prod} CPI moved "
            f"**{yoy:+.1f}%**."
        )
    if stats.get("is_volatile"):
        lines.append(
            "Note that this category has been **fairly volatile** recently. Single-month "
            "swings are common. Keeping a small buffer stock and staying flexible on brands or "
            "sources helps absorb sudden spikes."
        )

    lines.append("#### What you can do")
    lines.append(_trend_sentence(stats))
    tip = PRODUCT_FORECAST_TIPS.get(product)
    if tip:
        lines.append(tip)

    lines.append(
        "_For planners and policymakers, the same signal points to acting ahead of the trend: "
        "for example timing buffer-stock releases, easing supply bottlenecks or targeting support "
        "to the households most affected._"
    )
    lines.append(
        "> **Note:** This is an automated, educational interpretation of a statistical forecast, "
        "not financial, investment, or economic advice. Forecasts are estimates and can be wrong; "
        "use your own judgement for important decisions."
    )
    return "\n\n".join(lines)


def _stats_for_prompt(stats):
    """Compact, numbers-only summary for the AI helper (contains no personal data)."""
    keys = [
        "product", "sector", "forecast_value", "forecast_date", "pct_above_base",
        "purchasing_power_of_100", "last_obs_value", "last_obs_date", "abs_change",
        "pct_change", "horizon_months", "annualized_pct", "yoy_pct", "volatility_pct",
        "forecast_low", "forecast_high", "direction", "strength", "is_volatile",
    ]
    out = {}
    for k in keys:
        v = stats.get(k)
        out[k] = round(v, 2) if isinstance(v, float) else v
    return json.dumps(out, ensure_ascii=False)


@st.cache_data(show_spinner=False)
def ai_interpret_cpi(stats_json, language):
    """Optional AI-tailored advice, built ONLY from the derived numbers."""
    if not GEMINI_AVAILABLE or not key_looks_valid(get_api_key()):
        return None
    prompt = (
        "You are explaining an Indian CPI (Consumer Price Index) forecast to an everyday reader. "
        "You are given ONLY pre-computed numbers as JSON (no personal data). Using just these "
        "numbers, write a short, warm, practical explanation in at most 180 words.\n\n"
        "Cover: (1) what the forecast CPI value means relative to the base period of 100; "
        "(2) whether prices are expected to rise, fall, or stay steady, and by roughly how much; "
        "(3) two or three concrete, realistic actions a household can take for THIS product "
        "category and trend. Do not invent any numbers beyond those given, do not output JSON, "
        "and do not add a preamble.\n"
        f"Write the answer in {language}.\n\n"
        f"NUMBERS:\n{stats_json}"
    )
    return _gemini_generate(prompt)


def render_cpi_interpretation(stats, context_key, button_label="Interpret this graph"):
    """Render the 'Know more' button and, on demand, the interpretation panel."""
    show_flag = f"show_interp::{context_key}"
    if show_flag not in st.session_state:
        st.session_state[show_flag] = False

    if st.button("📘 " + button_label, key=f"interp_btn::{context_key}"):
        st.session_state[show_flag] = True

    if not st.session_state[show_flag]:
        return

    with st.container():
        st.divider()
        st.markdown(build_cpi_interpretation_text(stats))

        if ai_ready():
            st.divider()
            st.caption(
                "Optional: generate AI-tailored advice for this exact forecast. Only the chart's "
                "numbers are sent for processing, never your receipts or personal data."
            )
            lang = st.selectbox(
                "Advice language",
                ["English", "Hindi", "Telugu"],
                key=f"interp_lang::{context_key}",
            )
            if st.button("✨ Get AI-tailored advice", key=f"interp_ai_btn::{context_key}"):
                with st.spinner("Preparing tailored advice..."):
                    st.session_state[f"interp_ai_result::{context_key}"] = ai_interpret_cpi(
                        _stats_for_prompt(stats), lang
                    )

            ai_state = st.session_state.get(f"interp_ai_result::{context_key}")
            if isinstance(ai_state, str) and ai_state.startswith("__ERROR__"):
                st.info(
                    "AI advice is unavailable right now (" + ai_state[len("__ERROR__"):] + "). "
                    "The summary above still applies."
                )
            elif isinstance(ai_state, str) and ai_state:
                st.markdown("**AI-tailored advice**")
                st.markdown(ai_state)


# ============================================================================
# SHARED PRICE DATASET  (Google Sheets)  +  CATEGORISATION
# ============================================================================
# Receipt line items are sorted into the SAME categories as the inflation
# graphs (Fruits, Vegetables, Spices, Fuel and light, Health). Anything that
# does not match goes to "Other". The categorised rows are written to a shared
# Google Sheet, kept newest-first, de-duplicated so the sheet stays compact,
# and can be read back on the forecast page as a real-world community signal.
#
# Categorisation runs locally (no API key, no internet, no quota cost).
#
# CREDENTIALS: a Google service-account key is needed to write to the sheet.
#   - On Streamlit Cloud (deployed from GitHub): paste the key JSON into the
#     app's Secrets as a [gcp_service_account] table.
#   - Running locally: place the key file at
#     <Resources>/ayara_service_account.json  (or set AYARA_GCP_CREDENTIALS).
#   Then SHARE the Google Sheet with the service account's client_email
#   (Editor access), or writes will be refused by Google.

RECEIPTS_SHEET_ID = "15xGoX-8K7AhGWulx0SNoSwU1GkphAmi4ISwLBVtpXzM"
RECEIPTS_SHEET_TAB = "Receipts"
SHEET_HEADER = [
    "date", "category", "item", "quantity",
    "unit_price", "line_total", "currency", "merchant", "logged_at",
]
GSHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Categories receipts can fall into, in priority order. "Other" is the catch-all.
CATEGORY_PRIORITY = ["Health", "Fuel and light", "Spices", "Fruits", "Vegetables"]

# Keyword banks (Indian grocery context). Matching is whole-word and
# case-insensitive. Order matters: more specific buckets are checked first so,
# e.g., "chilli powder" -> Spices while bare "green chilli" -> Vegetables.
CATEGORY_KEYWORDS = {
    "Health": [
        "medicine", "medicines", "tablet", "tablets", "capsule", "capsules",
        "syrup", "paracetamol", "crocin", "dolo", "calpol", "antibiotic",
        "amoxicillin", "azithromycin", "vitamin", "vitamins", "multivitamin",
        "supplement", "ointment", "bandage", "band aid", "dettol", "savlon",
        "antiseptic", "mask", "masks", "sanitizer", "thermometer", "syringe",
        "injection", "insulin", "ors", "cough syrup", "painkiller", "ibuprofen",
        "aspirin", "antacid", "digene", "eno", "gelusil", "pantoprazole",
        "omeprazole", "vicks", "iodex", "moov", "volini", "balm", "inhaler",
        "asthalin", "glucometer", "test strips", "bp monitor", "pharmacy",
        "pharma", "chemist", "medical store", "prescription", "gauze",
        "eye drops", "ear drops", "lozenges", "strepsils", "betadine",
        "zincovit", "becosules", "shelcal", "calcium tablet", "iron tablet",
        "electral", "pedialyte", "cetirizine", "dolo 650", "saline",
    ],
    "Fuel and light": [
        "petrol", "diesel", "lpg", "gas cylinder", "cylinder", "cng",
        "kerosene", "electricity", "electricity bill", "power bill",
        "current bill", "light bill", "fuel", "gasoline", "firewood",
        "charcoal", "coal", "candle", "candles", "matchbox", "match box",
        "matches", "led bulb", "tube light", "tubelight", "cfl", "lantern",
        "torch", "led light", "bulb",
    ],
    "Spices": [
        "turmeric", "turmeric powder", "haldi", "cumin", "jeera",
        "coriander powder", "coriander seeds", "dhania", "dhaniya",
        "chilli powder", "chili powder", "red chilli powder", "lal mirch",
        "mirchi powder", "mustard", "mustard seeds", "rai", "sarson",
        "fennel", "saunf", "fenugreek seeds", "methi seeds", "fenugreek powder",
        "cardamom", "elaichi", "clove", "cloves", "laung", "cinnamon",
        "dalchini", "black pepper", "kali mirch", "pepper powder", "peppercorn",
        "bay leaf", "tej patta", "asafoetida", "hing", "nutmeg", "jaiphal",
        "mace", "javitri", "star anise", "garam masala", "masala",
        "chaat masala", "sambar powder", "rasam powder", "biryani masala",
        "saffron", "kesar", "tamarind", "imli", "poppy seeds", "khus khus",
        "sesame seeds", "carom", "ajwain", "dry ginger", "sonth", "salt",
        "namak", "asafetida",
    ],
    "Fruits": [
        "apple", "apples", "banana", "bananas", "mango", "mangoes", "orange",
        "oranges", "grape", "grapes", "guava", "papaya", "pomegranate", "anar",
        "watermelon", "muskmelon", "melon", "pineapple", "lemon", "sweet lime",
        "mosambi", "musambi", "sapota", "chikoo", "chiku", "pear", "peach",
        "plum", "kiwi", "strawberry", "strawberries", "jackfruit",
        "custard apple", "sitaphal", "fig", "anjeer", "dates", "khajur",
        "cherry", "cherries", "litchi", "lychee", "blueberry", "raspberry",
        "dragon fruit", "coconut", "fruit",
    ],
    "Vegetables": [
        "tomato", "tomatoes", "onion", "onions", "pyaz", "potato", "potatoes",
        "aloo", "brinjal", "eggplant", "baingan", "okra", "ladies finger",
        "bhindi", "cauliflower", "gobi", "gobhi", "cabbage", "patta gobi",
        "carrot", "carrots", "gajar", "beans", "french beans", "peas", "matar",
        "green peas", "capsicum", "shimla mirch", "bell pepper", "spinach",
        "palak", "methi leaves", "fenugreek leaves", "coriander leaves",
        "cilantro", "cucumber", "kheera", "bottle gourd", "lauki", "ghiya",
        "ridge gourd", "turai", "bitter gourd", "karela", "pumpkin", "kaddu",
        "radish", "mooli", "beetroot", "beet", "drumstick", "ginger", "adrak",
        "garlic", "lehsun", "green chilli", "green chili", "hari mirch",
        "mirchi", "mushroom", "mushrooms", "sweet corn", "corn", "bhutta",
        "yam", "colocasia", "arbi", "turnip", "shalgam", "lettuce", "broccoli",
        "zucchini", "spring onion", "raw banana", "plantain", "sweet potato",
        "shakarkandi", "ash gourd", "snake gourd", "cluster beans", "gavar",
        "tinda", "parwal", "curry leaves", "kadi patta", "leek", "vegetable",
    ],
}

# Pre-compile one regex per category for fast, whole-word matching.
_CATEGORY_REGEX = {
    cat: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(set(kws), key=len, reverse=True)) + r")\b"
    )
    for cat, kws in CATEGORY_KEYWORDS.items()
}


def categorize_item(description):
    """Map a receipt item description to one of the inflation categories.

    Returns the category name, or "Other" if nothing matches. Pure/offline.
    """
    if description is None:
        return "Other"
    text = str(description).lower()
    if not text.strip():
        return "Other"
    for cat in CATEGORY_PRIORITY:
        if _CATEGORY_REGEX[cat].search(text):
            return cat
    return "Other"


def _norm_date(v):
    """Normalise a date to 'YYYY-MM-DD' so a text sort = newest-first."""
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", "null", "nat"):
        return ""
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s


def _num_or_blank(v):
    """Return a clean number (int when whole) or '' so the sheet stays tidy."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        f = float(v)
        return int(f) if f == int(f) else round(f, 2)
    except Exception:
        return "" if v is None else str(v).strip()


def _load_service_account_info():
    """Return (service_account_dict, source) or (None, None)."""
    # 1) Streamlit secrets (works on GitHub-deployed Streamlit Cloud apps).
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"]), "secrets"
    except Exception:
        pass
    # 2) Local key file (the "secret file" when running on your own machine).
    path = os.environ.get("AYARA_GCP_CREDENTIALS") or os.path.join(
        BASE_DIR, "ayara_service_account.json"
    )
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh), path
        except Exception:
            return None, None
    return None, None


@st.cache_resource(show_spinner=False)
def _get_gspread_client():
    """Authorised gspread client, or None if unavailable/misconfigured."""
    if not GSPREAD_AVAILABLE:
        return None
    info, _ = _load_service_account_info()
    if not info:
        return None
    try:
        creds = Credentials.from_service_account_info(info, scopes=GSHEET_SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None


def sheets_ready():
    """True when the app can talk to Google Sheets."""
    return _get_gspread_client() is not None


def _open_receipts_ws():
    """Open (creating if needed) the receipts worksheet, header guaranteed."""
    client = _get_gspread_client()
    if client is None:
        raise RuntimeError(
            "Google Sheets is not connected. Add a service-account key and share "
            "the sheet with its client_email."
        )
    sh = client.open_by_key(RECEIPTS_SHEET_ID)
    try:
        ws = sh.worksheet(RECEIPTS_SHEET_TAB)
    except Exception:
        ws = sh.add_worksheet(
            title=RECEIPTS_SHEET_TAB, rows=1000, cols=len(SHEET_HEADER)
        )
        ws.update(range_name="A1", values=[SHEET_HEADER])
        return ws
    # Ensure the header row is present and correct.
    try:
        first = [c.strip().lower() for c in ws.row_values(1)]
    except Exception:
        first = []
    if first[: len(SHEET_HEADER)] != SHEET_HEADER:
        ws.update(range_name="A1", values=[SHEET_HEADER])
    return ws


def _row_key(date_val, item, line_total, merchant):
    """Stable de-dup key for one stored line item."""
    return "|".join([
        str(date_val).strip().lower(),
        str(item).strip().lower(),
        str(line_total).strip(),
        str(merchant).strip().lower(),
    ])


def save_receipts_to_sheet(df, keep_other=True):
    """Categorise, de-dupe, append and re-sort (newest-first).

    Returns a summary dict: appended, skipped_dupes, skipped_other, counts.
    """
    ws = _open_receipts_ws()
    existing = ws.get_all_values()

    existing_keys = set()
    for r in existing[1:]:  # skip header
        try:
            existing_keys.add(_row_key(r[0], r[2], r[5], r[7]))
        except Exception:
            continue

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []
    counts = {c: 0 for c in (PRODUCT_LIST + ["Other"])}
    skipped_dupes = 0
    skipped_other = 0

    for _, row in df.iterrows():
        desc = row.get("description")
        cat = categorize_item(desc)
        if cat == "Other" and not keep_other:
            skipped_other += 1
            continue
        date_val = _norm_date(row.get("date"))
        item = "" if desc is None else str(desc).strip()
        qty = _num_or_blank(row.get("quantity"))
        unit_price = _num_or_blank(row.get("unit_price"))
        line_total = _num_or_blank(row.get("line_total"))
        currency = "" if row.get("currency") is None else str(row.get("currency")).strip()
        merchant = "" if row.get("merchant") is None else str(row.get("merchant")).strip()

        key = _row_key(date_val, item, line_total, merchant)
        if key in existing_keys:
            skipped_dupes += 1
            continue
        existing_keys.add(key)
        new_rows.append(
            [date_val, cat, item, qty, unit_price, line_total, currency, merchant, stamp]
        )
        counts[cat] += 1

    appended = 0
    if new_rows:
        ws.append_rows(new_rows, value_input_option="USER_ENTERED")
        appended = len(new_rows)
        # Keep the sheet newest-first: sort data rows by date desc, then time desc.
        try:
            last_row = len(existing) + appended
            ws.sort((1, "des"), (9, "des"), range=f"A2:I{last_row}")
        except Exception:
            pass  # data is saved even if the optional re-sort is unsupported

    return {
        "appended": appended,
        "skipped_dupes": skipped_dupes,
        "skipped_other": skipped_other,
        "counts": counts,
    }


@st.cache_data(ttl=300, show_spinner=False)
def _read_sheet_records():
    """Read the shared sheet into a DataFrame (cached 5 min). None on failure."""
    client = _get_gspread_client()
    if client is None:
        return None
    try:
        sh = client.open_by_key(RECEIPTS_SHEET_ID)
        ws = sh.worksheet(RECEIPTS_SHEET_TAB)
        values = ws.get_all_values()
    except Exception:
        return None
    if not values or len(values) < 2:
        return pd.DataFrame(columns=SHEET_HEADER)
    header = [c.strip().lower() for c in values[0]]
    return pd.DataFrame(values[1:], columns=header)


def crowd_price_signal(product, lookback_days=60):
    """Recent community price signal for a category, from the shared sheet.

    Compares the latest `lookback_days` window against the prior window of the
    same length. Returns a dict (possibly with n=0) or None if the sheet is
    unreadable.
    """
    df = _read_sheet_records()
    if df is None or df.empty or "category" not in df.columns:
        return None
    sub = df[df["category"] == product].copy()
    base = {"product": product, "n": 0, "lookback_days": lookback_days}
    if sub.empty:
        return base

    sub["_date"] = pd.to_datetime(sub.get("date"), errors="coerce")
    sub["_price"] = pd.to_numeric(sub.get("unit_price"), errors="coerce")
    sub = sub.dropna(subset=["_date"])
    if sub.empty:
        base["n"] = int(df[df["category"] == product].shape[0])
        return base

    latest = sub["_date"].max()
    recent_start = latest - pd.Timedelta(days=lookback_days)
    prev_start = latest - pd.Timedelta(days=2 * lookback_days)
    recent = sub[sub["_date"] > recent_start]
    prev = sub[(sub["_date"] > prev_start) & (sub["_date"] <= recent_start)]

    recent_avg = recent["_price"].mean()
    prev_avg = prev["_price"].mean()
    out = {
        "product": product,
        "n": int(sub.shape[0]),
        "n_recent": int(recent.shape[0]),
        "recent_avg": None if pd.isna(recent_avg) else float(recent_avg),
        "prev_avg": None if pd.isna(prev_avg) else float(prev_avg),
        "latest_date": latest.strftime("%d %b %Y"),
        "lookback_days": lookback_days,
        "pct_change": None,
        "direction": None,
    }
    if (
        out["recent_avg"] is not None
        and out["prev_avg"] not in (None, 0)
        and not pd.isna(prev_avg)
        and prev_avg
    ):
        pct = (recent_avg - prev_avg) / prev_avg * 100.0
        out["pct_change"] = float(pct)
        out["direction"] = "rising" if pct > 1 else "falling" if pct < -1 else "broadly flat"
    return out


# --- Community-adjusted ("blended") forecast tuning -------------------------
# The model forecast is always the base. The community price trend can nudge it
# up or down, but only within a capped band, and weighted by how much recent
# receipt data exists, so a handful of receipts can never swing the forecast.
BLEND_MAX_ADJ_PCT = 5.0        # hard cap: never move the forecast more than ±5%
BLEND_FULL_CONF_N = 20         # this many recent receipt items => full weight
BLEND_MIN_RECEIPTS = 5         # CAPACITY GATE: a category needs at least this many
                               # recent receipts before ANY trend is measured/shown
                               # (raise to 10 for a stricter bar)
BLEND_MIN_SIGNAL_PCT = 1.0     # ignore community moves smaller than this (noise)


def compute_community_adjustment(product):
    """Turn the community price signal into a capped, confidence-weighted
    percentage nudge for the model forecast.

    Capacity rule: a category must have at least ``BLEND_MIN_RECEIPTS`` recent
    receipts before a trend is even measured. Below that there simply isn't
    enough data for a meaningful trend. The model forecast is then shown as-is.

    Returns a dict. ``applied`` is False (and ``adj_pct`` is 0.0) whenever there
    is not enough clean signal to adjust. This function never raises.
    """
    out = {
        "applied": False, "adj_pct": 0.0, "raw_pct": None, "n_recent": 0,
        "n_total": 0, "direction": None, "weight": 0.0, "reason": "",
        "lookback_days": 60, "min_receipts": BLEND_MIN_RECEIPTS,
    }
    try:
        sig = crowd_price_signal(product)
    except Exception:
        out["reason"] = "unavailable"
        return out
    if not sig or sig.get("n", 0) == 0:
        out["reason"] = "no_data"
        return out
    out["lookback_days"] = sig.get("lookback_days", 60)
    out["n_total"] = int(sig.get("n", 0) or 0)
    raw = sig.get("pct_change")
    n_recent = int(sig.get("n_recent", 0) or 0)
    out["raw_pct"] = raw
    out["n_recent"] = n_recent
    out["direction"] = sig.get("direction")
    # Capacity gate FIRST: not enough recent receipts => no trend at all.
    if n_recent < BLEND_MIN_RECEIPTS:
        out["reason"] = "too_few_recent"
        return out
    if raw is None:
        out["reason"] = "no_recent_trend"
        return out
    if abs(raw) < BLEND_MIN_SIGNAL_PCT:
        out["reason"] = "signal_too_small"
        return out
    weight = min(1.0, n_recent / float(BLEND_FULL_CONF_N))
    capped = max(-BLEND_MAX_ADJ_PCT, min(BLEND_MAX_ADJ_PCT, raw))
    out["weight"] = weight
    out["adj_pct"] = capped * weight
    out["applied"] = abs(out["adj_pct"]) > 1e-9
    if not out["applied"]:
        out["reason"] = "signal_too_small"
    return out


def render_crowd_signal(product, model_direction=None, context_key=""):
    """Button + panel surfacing the community price signal next to a forecast."""
    if not GSPREAD_AVAILABLE:
        return
    flag = f"show_crowd::{context_key}"
    if st.button(
        "📊 Community price signal (from shared receipts)",
        key=f"crowd_btn::{context_key}",
    ):
        st.session_state[flag] = True
    if not st.session_state.get(flag):
        return

    with st.container():
        st.divider()
        if not sheets_ready():
            st.info(
                "Connect Google Sheets (add a service-account key) to see live "
                "community price signals here."
            )
            return
        sig = crowd_price_signal(product)
        if sig is None:
            st.info("The shared price sheet could not be read right now.")
            return
        if sig.get("n", 0) == 0:
            st.caption(
                f"No community receipts logged for **{product}** yet. Be the first "
                "by uploading receipts under *Crowdsourcing Receipts*."
            )
            return

        st.markdown(f"**Community price signal: {product}**")
        c = st.columns(3)
        c[0].metric("Receipts logged", sig["n"])
        if sig.get("recent_avg") is not None:
            c[1].metric(
                f"Avg item price (last {sig['lookback_days']}d)",
                f"₹{sig['recent_avg']:.1f}",
            )
        if sig.get("pct_change") is not None:
            c[2].metric("Recent vs prior", f"{sig['pct_change']:+.1f}%")

        if sig.get("direction"):
            msg = f"Community receipt prices for {product.lower()} are **{sig['direction']}** recently"
            if sig.get("pct_change") is not None:
                msg += f" ({sig['pct_change']:+.1f}% vs the previous {sig['lookback_days']} days)"
            msg += "."
            if model_direction in ("rising", "falling") and sig["direction"] in ("rising", "falling"):
                if model_direction == sig["direction"]:
                    msg += f" This **agrees** with the model's {model_direction} forecast, which adds confidence."
                else:
                    msg += (
                        f" This **differs** from the model's {model_direction} forecast: worth watching, "
                        "as on-the-ground prices may be moving differently."
                    )
            st.markdown(msg)

        st.caption(
            "This is a real-world signal from community-submitted receipts. It **contextualises** the "
            "forecast and does not retrain the underlying model. Receipt values are rupee prices per item "
            "while the forecast is a CPI index, so compare direction and momentum rather than absolute numbers."
        )


# ============================================================================
# PRIVACY POLICY & TERMS AND CONDITIONS
# ============================================================================
# IMPORTANT: This is a thorough, good-faith template. It is NOT legal advice and
# is not a guarantee of compliance. Before any public or commercial release, have
# a qualified attorney review and adapt it for your jurisdiction(s), your actual
# data flows, and any third-party services you enable.
PRIVACY_LAST_UPDATED = "July 13, 2026"
PRIVACY_OPERATOR = "Ayara"
PRIVACY_CONTACT_EMAIL = "aindukuri@wisc.edu"
PRIVACY_JURISDICTION = "India"

PRIVACY_POLICY_MD = f"""
## Privacy Policy

**Last updated: {PRIVACY_LAST_UPDATED}**

This Privacy Policy describes how {PRIVACY_OPERATOR} ("we", "us", or "our") collects, uses, discloses, and protects information in connection with the Ayara application and its features (collectively, the "Service"), and the choices and rights you have. By accessing or using the Service, you acknowledge that you have read and understood this Policy. If you do not agree, do not use the Service.

### 1. Summary
- We process the receipts and files you choose to upload only to read their contents (merchant, date, totals, taxes, and line items) and show the results back to you.
- By default, uploaded files and extracted data are held only in memory for your current session and are **not** stored on our servers after the session ends, unless you explicitly download them or an operator has enabled optional cloud logging.
- To read files automatically, file contents are transmitted to **Google's Gemini API** for processing. We do **not** sell your personal information.
- You can use most of the Service without providing personal data, and you can withdraw consent at any time by not uploading files.

### 2. Information we collect
**(a) Information you provide.** Receipt and invoice images, screenshots and PDF files that you upload and any information contained in them, which may include merchant names, dates, item descriptions, quantities, prices, totals and taxes and occasionally personal details printed on a receipt (for example a loyalty name, card last-four digits or address). Please avoid uploading documents containing sensitive personal data about yourself or others. Redact such details where reasonably possible.

**(b) Location context (optional).** Where the Service associates prices with a place, it relies on store information, document metadata, or details you choose to enter. The Service is designed to avoid intrusive or continuous GPS tracking.

**(c) Usage and technical data.** Limited operational data needed to run the Service, such as a local counter of AI requests made per day (stored on your own device to enforce rate limits; see Section 10) and standard information your browser or hosting environment generates (for example, error logs). The exact technical data depends on how and where the Service is deployed.

**(d) We do not intentionally collect** government identifiers, full payment-card numbers, biometric data, or special-category data. Please do not submit such information.

### 3. How we use information
We use information to: (i) read and structure your uploaded receipts; (ii) display extracted results and let you download them; (iii) generate aggregate, non-identifying price and inflation insights; (iv) operate, secure, maintain, and improve the Service; (v) enforce usage limits and prevent abuse; and (vi) comply with legal obligations.

### 4. Legal bases (EEA / UK users)
Where the GDPR or UK GDPR applies, we rely on: your **consent** (Art. 6(1)(a)) for processing the receipts you upload and for optional features; **legitimate interests** (Art. 6(1)(f)) in operating, securing, and improving the Service in a way that does not override your rights; and **legal obligation** (Art. 6(1)(c)) where applicable. You may withdraw consent at any time; this does not affect processing already carried out.

### 5. AI processing and third-party services
To read documents automatically, file contents and prompts are sent to **Google's Gemini API** and processed on Google's infrastructure subject to Google's terms and privacy commitments. We do not control Google's processing; please review Google's applicable terms and privacy policy. If an operator enables optional **Google Sheets** logging (via the `gspread` integration), extracted data you submit may also be written to a Google Sheet controlled by that operator. We use third parties only to provide the Service and do not authorise them to use your content to train their models except as permitted by their applicable enterprise/API terms.

### 6. Data retention
By default, uploaded files and extracted data exist only in memory for the duration of your session and are discarded when the session ends or you close the app. Data you download is stored wherever you save it and is your responsibility. If optional cloud logging is enabled by an operator, retention is governed by that operator's configuration; we recommend retaining only what is necessary and applying a defined retention schedule.

### 7. Disclosure of information
We do not sell personal information. We may disclose information: (i) to service providers/processors (such as Google) strictly to operate the Service; (ii) if required by law, regulation, legal process, or an enforceable governmental request; (iii) to protect the rights, property, safety, or security of users, the public, or us, including to prevent fraud or abuse; and (iv) in connection with a merger, acquisition, financing, or sale of assets, subject to this Policy.

### 8. International data transfers
Processing (including by Google) may occur in countries other than your own, whose data-protection laws may differ. Where required, we rely on appropriate safeguards such as Standard Contractual Clauses or equivalent mechanisms for cross-border transfers.

### 9. Your rights and choices
Depending on your location, you may have rights to access, correct, update, or delete your personal data; to restrict or object to certain processing; to data portability; and to withdraw consent.

**EEA / UK (GDPR):** you also have the right to lodge a complaint with your local supervisory authority.

**California (CCPA/CPRA):** you have rights to know, access, correct, and delete personal information, to opt out of the "sale" or "sharing" of personal information, and to limit the use of sensitive personal information. We do not sell or share personal information as those terms are defined, and we will not discriminate against you for exercising your rights.

Because the default design keeps your data only within your session, the most direct way to exercise deletion is to close the session and delete any files you downloaded. For other requests, contact us using Section 13.

### 10. Cookies and local storage
The Service does not require advertising cookies. It may keep a small local file on your own device named `.ayara_gemini_usage.json` (in your home directory) that records only a date and a count of AI requests, solely to enforce daily usage limits. You can delete this file at any time; it contains no receipt contents or personal data.

### 11. Security
We apply reasonable technical and organisational measures appropriate to the nature of the data, including data minimisation, session-scoped handling, upload size/type limits, and transmission to processors over encrypted connections. No method of transmission or storage is perfectly secure, and we cannot guarantee absolute security.

### 12. Children's privacy
The Service is not directed to children. We do not knowingly collect personal data from children under the age required by your jurisdiction (for example, under 13 under the U.S. COPPA, or under 16 in parts of the EEA). If you believe a child has provided personal data, contact us so we can take appropriate action.

### 13. Changes and contact
We may update this Policy from time to time; the "Last updated" date reflects the latest version, and material changes will be indicated within the Service where feasible. Questions or requests: **{PRIVACY_CONTACT_EMAIL}** ({PRIVACY_OPERATOR}).
"""

TERMS_CONDITIONS_MD = f"""
## Terms and Conditions

**Last updated: {PRIVACY_LAST_UPDATED}**

These Terms and Conditions ("Terms") govern your access to and use of the Ayara application and its features (the "Service") provided by {PRIVACY_OPERATOR}. By accessing or using the Service, you agree to be bound by these Terms and by the Privacy Policy. If you do not agree, do not use the Service.

### 1. Eligibility
You must be at least 16 years old (or the age of digital consent in your jurisdiction, if higher) and capable of forming a binding contract. If you use the Service on behalf of an organisation, you represent that you are authorised to bind it.

### 2. The Service
Ayara lets you upload receipts and invoices, uses automated optical character recognition and AI to extract their contents, and presents price and inflation information and forecasts for educational and informational purposes. Features, availability, and accuracy may change at any time.

### 3. Your content and responsibilities
You retain ownership of files you upload ("User Content"). You represent and warrant that you have the right to upload and process each file and that doing so does not violate any law or third-party right. You agree to remove or redact other people's personal information where reasonably possible, and not to upload unlawful, infringing, or sensitive content (for example, full payment-card numbers or government identifiers). You grant us a limited, non-exclusive, worldwide, royalty-free licence to host, process, transmit (including to processors such as Google), and display User Content solely to provide the Service to you and, in aggregated and de-identified form, to generate price and inflation insights.

### 4. Acceptable use
You agree not to: (i) use the Service for any unlawful, harmful, or fraudulent purpose; (ii) attempt to gain unauthorised access to the Service or its systems; (iii) interfere with or disrupt the Service, circumvent usage limits, or impose an unreasonable load; (iv) reverse engineer or misuse any AI feature; (v) upload malware or harmful code; or (vi) use the Service to infringe intellectual-property or privacy rights.

### 5. AI output and no professional advice
AI and forecasting features are automated and may be inaccurate, incomplete, or outdated. The CPI interpretations, price analyses, and forecasts are **for general educational and informational purposes only** and do **not** constitute financial, investment, accounting, tax, legal, economic, or other professional advice. They are estimates based on historical data and statistical models and should not be solely relied upon. You are solely responsible for the decisions you make, and you should consult a qualified professional before acting.

### 6. Intellectual property
The Service, including its software, design, and the content we provide (excluding User Content and third-party materials), is owned by {PRIVACY_OPERATOR} or its licensors and is protected by intellectual-property laws. Except as expressly permitted, you may not copy, modify, distribute, sell, or create derivative works from the Service.

### 7. Third-party services
The Service relies on third-party services (including Google's Gemini API and, optionally, Google Sheets). Your use of those services may be subject to their own terms and policies. We are not responsible for third-party services, and their availability or terms may change.

### 8. Disclaimer of warranties
THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, WHETHER EXPRESS, IMPLIED, OR STATUTORY, INCLUDING IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, ACCURACY, AND NON-INFRINGEMENT. WE DO NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED, ERROR-FREE, OR SECURE, OR THAT RESULTS OR FORECASTS WILL BE ACCURATE OR RELIABLE.

### 9. Limitation of liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, {PRIVACY_OPERATOR} AND ITS AFFILIATES, OFFICERS, EMPLOYEES, AND AGENTS WILL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, OR FOR ANY LOSS OF PROFITS, REVENUE, DATA, OR GOODWILL, ARISING OUT OF OR RELATING TO YOUR USE OF (OR INABILITY TO USE) THE SERVICE, EVEN IF ADVISED OF THE POSSIBILITY. TO THE EXTENT LIABILITY CANNOT BE EXCLUDED, OUR TOTAL AGGREGATE LIABILITY WILL NOT EXCEED THE GREATER OF (A) THE AMOUNT YOU PAID US TO USE THE SERVICE IN THE 12 MONTHS BEFORE THE CLAIM, OR (B) USD 100. SOME JURISDICTIONS DO NOT ALLOW CERTAIN LIMITATIONS, SO SOME OF THE ABOVE MAY NOT APPLY TO YOU.

### 10. Indemnification
You agree to indemnify and hold harmless {PRIVACY_OPERATOR} and its affiliates from claims, damages, liabilities, and expenses (including reasonable legal fees) arising out of your User Content, your use of the Service, or your violation of these Terms or any law or third-party right.

### 11. Suspension and termination
We may suspend or terminate access to the Service, in whole or in part, at any time and for any reason, including for a violation of these Terms. You may stop using the Service at any time. Sections that by their nature should survive (including Sections 3, 5, 6, 8, 9, 10, and 12) will survive termination.

### 12. Governing law and disputes
These Terms are governed by the laws of {PRIVACY_JURISDICTION}, without regard to conflict-of-laws rules. The courts located in {PRIVACY_JURISDICTION} will have exclusive jurisdiction, unless mandatory local law provides otherwise. You agree to attempt to resolve disputes informally by contacting us first.

### 13. Changes to these Terms
We may modify these Terms from time to time. Changes are effective when posted (or as otherwise stated), and your continued use after changes means you accept them. If you do not agree, stop using the Service.

### 14. Miscellaneous
If any provision is found unenforceable, the remaining provisions remain in effect. Our failure to enforce a provision is not a waiver. These Terms and the Privacy Policy are the entire agreement between you and us regarding the Service and supersede prior agreements on the subject.

### 15. Contact
Questions about these Terms: **{PRIVACY_CONTACT_EMAIL}** ({PRIVACY_OPERATOR}).
"""

DATA_SUMMARY_MD = """
### Data handling at a glance

| Topic | How Ayara handles it |
|---|---|
| What you upload | Receipt / invoice images, screenshots, or PDFs you choose |
| What is extracted | Merchant, date, currency, totals, taxes, and line items |
| Where it goes | File contents are sent to Google's Gemini API to be read |
| How long it's kept | In-session only by default; gone when the session ends |
| Storage on our servers | None by default (optional operator-enabled logging aside) |
| Selling your data | Never |
| Location tracking | No intrusive/continuous GPS; place is inferred from the receipt or your input |
| Local file on your device | `.ayara_gemini_usage.json`: only a date + request count, for rate limits |
| Your controls | Don't upload to withhold data; download to keep; close the session to discard |

Please redact other people's personal details before uploading, and avoid submitting card numbers or government IDs.
"""


def render_privacy_and_terms():
    st.title("Privacy Policy & Terms and Conditions")
    st.caption(f"Last updated: {PRIVACY_LAST_UPDATED}")
    st.warning(
        "Template notice: This Privacy Policy and these Terms are a thorough, good-faith "
        "template, not legal advice and not a guarantee of compliance. Before any public or "
        "commercial release, have a qualified attorney review and adapt them for your "
        "jurisdiction(s), your actual data flows and any third-party services you enable."
    )
    tab1, tab2, tab3 = st.tabs(["Privacy Policy", "Terms & Conditions", "Data Summary"])
    with tab1:
        st.markdown(PRIVACY_POLICY_MD)
    with tab2:
        st.markdown(TERMS_CONDITIONS_MD)
    with tab3:
        st.markdown(DATA_SUMMARY_MD)
        st.download_button(
            "Download Privacy Policy & Terms (Markdown)",
            data=(PRIVACY_POLICY_MD + "\n\n---\n\n" + TERMS_CONDITIONS_MD).encode("utf-8"),
            file_name="Ayara_Privacy_and_Terms.md",
            mime="text/markdown",
        )


page = st.sidebar.radio(
    "Page Navigation",
    ["Home", "Analysis", "Inflation Forecasting", "Crowdsourcing Receipts", "Privacy & Terms"],
)

st.sidebar.divider()
if ai_ready():
    _used, _cap = quota_status()
    st.sidebar.caption(f"AI features (Gemini): enabled · {_used}/{_cap} requests used today")
elif GEMINI_AVAILABLE:
    st.sidebar.caption("AI features (Gemini): add GEMINI_API_KEY to enable")
else:
    st.sidebar.caption("AI features (Gemini): run  pip install google-genai  to enable")

st.sidebar.divider()
st.sidebar.caption(
    "🔒 Your privacy: uploaded receipts are processed for this session only and are "
    "not stored by Ayara. See **Privacy & Terms** in the menu above."
)



if page == "Home":
    st.title("Ayara: Your local market decoded. ")

    safe_image(LOGO_PATH, caption="Ayara Logo")

    st.header("Project Overview")
    st.write(""" Ayara is a Python and Streamlit application that brings Indian Consumer Price Index data, time series forecasts, multilingual explanations and receipt extraction into one interface. Its current analytical scope is five spending categories across national rural, urban and combined sectors. I trained and integrated 15 forecasting models, one for each category and sector combination. The application loads those previously trained models to generate forecasts and model supplied intervals.

A separate receipt workflow uses Gemini to extract structured information from images and PDFs, assigns items to categories through local text rules and offers CSV export and optional contribution to Google Sheets. An experimental layer compares receipt price averages across two periods and applies a capped adjustment beside the original forecast. This layer does not retrain the models or constitute a geographically resolved inflation index. The proposed local index is a separate research objective. """)

    st.divider()

    st.header("About Me")
    st.write("My name is Anand Varma Indukuri and I am a Computer Science and Data Science student at the University of Wisconsin Madison, class of 2029. Growing up in India, I often witnessed the quiet ways inflation shaped daily life. Families would adjust their meals, postpone purchases or stretch their budgets without fully realizing how much their expenses had grown. Official statistics rarely captured these everyday struggles, creating a gap between the numbers on paper and the reality people experienced. I wanted to create something that could close that gap by giving every household, regardless of income, a clear and honest picture of its cost of living. This project is also intended for those in government who make critical decisions, offering them precise and timely data from the ground. For me, it is not only a technical challenge but also a way to bring transparency, empowerment and a stronger voice to communities.")



if page == "Analysis":
    st.title("Analysis of India's Inflation by Category")
    safe_image(INFLATION_IMG_PATH, caption="Inflation")

    df = get_data()

    st.header("Data:")
    col1, col2 = st.columns([3, 1])

    products_selected = col1.multiselect("Select a product category", PRODUCT_LIST)
    sector_selected = col2.selectbox("Select sector", ["Rural", "Urban", "Rural+Urban"], index=2)

    relevant_cols = ["Month_Year"] + products_selected

    
    if products_selected:
        df_filtered = df[df["Sector"] == sector_selected].copy()
        df_products = df_filtered[relevant_cols].copy()
        df_products.set_index("Month_Year", inplace=True)
        df_products.sort_index(inplace=True)

        fig = plt.figure(figsize=(12, 6))
        for col in products_selected:
            plt.plot(df_products.index, df_products[col], label=col)
        plt.title(f'Selected Product CPI Over Time: {", ".join(products_selected)}', fontsize=14)
        plt.ylabel("CPI")
        plt.xlabel("Date")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=90, fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select one or more product categories above to see the CPI chart.")

    st.header("Analysis Shown Below: ")
    col_aud, col_cat, col_lang = st.columns(3)

    user_list = ["Government", "Regular User"]
    audience = col_aud.radio("Which analysis do you want?", user_list, index=0)
    analysis_category = col_cat.radio("Pick a category to analyze", PRODUCT_LIST, index=0)
   
    language = col_lang.radio("Language for the analysis", ["English", "Hindi", "Telugu"], index=0)

    simplify = st.checkbox("Make it shorter and easier to read (AI-powered)", value=False)

    ANALYSIS_TEXT = {
    'Government': {
        'Fruits': """ 
    1. Long-Term Inflationary Trend
     The CPI for fruits has consistently increased from about 103 in January 2013 to about 213 in July 2025, the latest month in the data (Rural+Urban), a cumulative rise of about 107 percent. This pattern indicates structural inflation rather than short-term fluctuations, caused by persistent factors such as rising input costs, higher fuel prices, increased labor wages and inefficiencies in the supply chain. For the government, this trend signals the need for structural reforms in the fruit market. Measures could include expanding cold chain networks through the Mission for Integrated Development of Horticulture, enhancing market integration via the National Agriculture Market platform and incentivizing private investment in modern storage and distribution facilities. These steps would reduce wastage, lower supply costs and slow the rate of price increases, directly benefiting both producers and consumers while mitigating inflationary pressure on the broader economy.
 
    2. Seasonal Variation and Harvest Cycles
     The CPI trend for fruits displays recurring peaks and troughs that align with harvest and lean    periods. Prices tend to rise before harvest seasons when supply is scarce and drop when the market is saturated with fresh produce. For example, mango prices rise sharply in March and April before falling in June once the main crop enters the market. Apple prices in northern India increase between March and May as stored stocks diminish, followed by a decline in September and October during harvest. Citrus fruits such as oranges and sweet limes often see price increases during the monsoon when supply chains slow and availability decreases. Recognizing these cycles allows the government to act preemptively. Policy tools could include facilitating staggered planting to ensure more consistent supply throughout the year, releasing stored produce during shortage months, and enabling timely imports to stabilize domestic prices without harming local producers.
 
    3. Price Shocks and Market Disruptions
     Beyond predictable seasonal patterns, the CPI data shows sudden and significant price changes that point to market shocks. These can be caused by extreme weather, pest outbreaks, transportation strikes, or emergency policy decisions. During the COVID-19 lockdown in April 2020, for instance, fruit CPI in cities rose steeply due to restricted interstate movement, while rural areas experienced a glut and falling prices due to unsold produce. In 2015, unseasonal rainfall in Maharashtra damaged banana and grape crops, leading to rapid price escalations. To manage such situations, the government can establish a national real-time price and supply monitoring system integrated with meteorological data. This system could trigger rapid interventions such as emergency transportation subsidies, temporary reductions in import tariffs, and targeted distribution of essential fruits through the Public Distribution System to keep consumer prices stable and prevent farmer losses.
 
4. Urban and Rural Price Divergence
 While the aggregated CPI indicates general trends, the underlying data reveals significant price differences between rural and urban markets. Urban areas tend to have more stable fruit prices due to better infrastructure, cold storage facilities, and more reliable transport links. Rural areas, in contrast, experience sharper fluctuations, sometimes with sudden surges during shortages and steep declines during gluts. In rural Bihar in 2019, banana prices collapsed by over 40 percent in a matter of weeks due to excess supply and inadequate storage, while urban consumers in Patna saw no such reduction. Addressing these disparities requires government investment in rural cold storage hubs, improvements in rural transport networks through the Pradhan Mantri Gram Sadak Yojana, and strengthening farmer-producer organizations to connect rural supply directly to urban demand. These actions would stabilize rural prices and enhance farmer incomes while keeping urban markets supplied at consistent rates.
 
5. Contribution to Headline Inflation and Monetary Policy
 Fruits contribute significantly to the Food and Beverages category in the CPI basket, which has a strong influence on headline inflation. Persistent increases in fruit prices can raise overall CPI, prompting the Reserve Bank of India to adjust monetary policy by raising interest rates. This can slow economic growth and increase borrowing costs across sectors. In July 2023, retail inflation reached 7.44 percent, with perishables such as tomatoes, onions, and certain fruits playing a key role in the rise. For the government, monitoring fruit CPI can serve as an early warning for inflationary trends. Possible interventions include temporary easing of import restrictions for specific fruits from countries such as Thailand and Vietnam, subsidizing the transport of surplus produce from high-supply to high-demand regions, and reducing export incentives when domestic prices threaten to push overall inflation beyond acceptable limits.
 
6. Climate Change and Risk Forecasting
 The increasing intensity of CPI fluctuations in recent years suggests a growing influence of climate-related disruptions. Extreme rainfall, prolonged heatwaves, and hailstorms have all affected fruit yields, leading to sharp price changes. In 2023, apple production in Himachal Pradesh fell significantly due to reduced chill hours and unseasonal rains, causing prices in Delhi and Chandigarh to rise by over 20 percent within two months. To counter this risk, the government should integrate CPI data with weather forecasting and satellite-based crop monitoring to identify potential shortages well in advance. Investments in climate-resilient horticulture, micro-irrigation systems, and the promotion of drought- and heat-tolerant fruit varieties would strengthen supply resilience. Additionally, introducing index-based crop insurance schemes that provide payouts when fruit CPI exceeds a set threshold could protect both farmers and consumers from severe climate-driven price shocks. """,
        'Vegetables': """ Long-Term Inflationary Trend
 The CPI for vegetables has risen markedly from about 102 in January 2013 to about 212 in July 2025, the latest month in the data (Rural+Urban), a cumulative increase of about 107 percent. Urban markets show an even steeper climb, from about 103 to about 243 over the same span, while rural markets rise from about 102 to about 196. This sustained, multi-year increase indicates structural pressures in horticultural supply chains, including post-harvest losses, elevated logistics costs and vulnerability to fuel and wage increases. For the government, the implication is to prioritize structural interventions in the vegetable value chain: accelerate cold-chain and pack-house capacity under the Mission for Integrated Development of Horticulture, expand market integration via e-NAM with time-bound onboarding of APMCs and incentivize private investment in reefer transport and last-mile storage. These steps reduce spoilage, lower intermediation costs and dampen price pass-throughs, moderating headline inflation without suppressing farmgate incomes.


Seasonal Variation and Monsoon Sensitivity
 The CPI for vegetables exhibits pronounced seasonal amplitude, typically softening in late winter-early summer and rising through the monsoon and post-monsoon months. In your series, average levels are lower around March-May and higher from July through December, peaking most often in October-November. This pattern reflects harvest cycles for leafy and fruit vegetables and the heightened logistics frictions during monsoon months, when weather disruptions and road conditions constrain inflows to urban mandis. A policy response should blend supply smoothing and information flow: promote staggered sowing calendars through Krishi Vigyan Kendras for key staples like tomatoes, onions and potatoes; operationalize district-level buffer norms to release stocks during lean weeks; and deploy real-time mandi arrival dashboards to pre-emptively redirect surplus from low-price zones to tightening markets.


Price Shocks and Idiosyncratic Spikes
 Beyond seasonality, the dataset captures sharp year-on-year surges, notably in late 2019 and early 2020 when vegetable CPI spiked by more than 30 to 45 percent. These bursts align with onion-led shocks following late and excessive rains and localized flooding in major producing belts, as well as transient logistics disruptions near the onset of the pandemic. Such shocks are amplified in metros, where dependence on long-haul supply lines is higher. To manage these episodes, establish an integrated early-warning system that fuses IMD rainfall alerts, remote-sensing crop progress, and mandi inflow anomalies. This should trigger swift measures: temporary import duty adjustments or quota releases for onions and tomatoes, time-bound freight subsidies to clear chokepoints, and calibrated Strategic Buffer releases through NCCF/NAFED to stabilize retail prices while averting distress sales in producing districts.


Urban-Rural Divergence and Volatility
 Your data show a persistent urban premium over rural vegetable prices, on average around 10 index points, with occasional gaps exceeding 30 points during stress periods. Urban volatility is also higher, reflecting longer supply chains and faster demand transmission. Meanwhile, rural markets can face sudden gluts and collapses when storage and aggregation are inadequate. Policy should narrow this gap by scaling rural aggregation (FPOs/SHGs) linked to assured urban off-take, financing decentralized cold rooms at block level and improving rural road connectivity under PMGSY to reduce transit time variability. Standardizing transparent grading at source and widening direct-to-retail channels in cities will reduce multiple handling and compress urban markups.


Contribution to Headline Inflation and Policy Coordination
 Vegetables carry meaningful weight within the Food and Beverages basket and are a frequent driver of short-run inflation volatility. The late-2019 to early-2020 surges in your series coincide with periods when food inflation pushed headline CPI higher, complicating monetary policy. For fiscal and administrative instruments to complement monetary stance, the government can operationalize dynamic buffer norms for TOP (tomato-onion-potato) and a handful of highly elastic vegetables; enable rapid inter-state movement through green corridors during stress; and adopt sunset-bound export curbs only when retail inflation risks breaching tolerance bands. This coordinated toolkit helps contain second-round effects from food spikes into non-food inflation, preserving growth while anchoring expectations.


Climate Change and Risk Forecasting
 The increasing amplitude and frequency of vegetable CPI spikes in recent years within your data suggest rising climate sensitivity: excess rainfall, heat waves and unseasonal storms disrupt yields and storability, especially for perishables. A forward-looking approach should integrate crop-stage weather risk with market signals: scale micro-irrigation and protected cultivation for high-volatility vegetables, promote climate-resilient varieties through targeted breeder seed expansion and link index-based insurance to market outcomes by adding CPI-or mandi-price triggers for specified districts. Coupled with satellite-based acreage nowcasts and truck-GPS feeds for corridor monitoring, this would allow pre-positioning of buffers and targeted logistics support weeks before shortages manifest in retail prices. """,
        'Spices': """ Long-Term Inflationary Trend
The CPI for spices has shown a steady upward trajectory from about 103 in January 2013 to about 221 in July 2025, the latest month in the data (Rural+Urban), a cumulative increase of about 115 percent. This long-term rise reflects structural factors, including persistent supply-demand imbalances, increased export demand from international markets, higher cultivation costs and periodic shortages of key commodities like coriander, cumin and turmeric. For the government, this indicates the need for supply-side reforms in the spice sector: expanding spice park infrastructure under the Spices Board, strengthening linkages to domestic wholesale markets to reduce excessive dependence on a few trading hubs and enhancing farmer access to precision irrigation and mechanized post-harvest processing. These measures would reduce storage losses, stabilize prices and increase domestic availability without curtailing export potential.


Seasonal Variation and Harvest Patterns
 The CPI trend for spices exhibits mild but consistent seasonal variation, with slight softening during peak harvest months and firming during lean storage periods. For example, cumin and coriander prices tend to drop between March and May following the rabi harvest, while turmeric prices often firm up from July to September due to reduced arrivals in the monsoon season. Recognizing these cycles allows the government to intervene through calibrated stock releases from accredited spice warehouses, promotion of off-season cultivation where agronomically feasible, and targeted import facilitation for high-demand spices during lean months to prevent price spikes in domestic retail markets.


Price Shocks and External Disruptions
 Beyond predictable cycles, the dataset shows sudden spikes, such as the sharp rise in late 2016 and mid-2020, driven by factors like unseasonal rains damaging standing crops, pest infestations or surges in export orders. During the COVID-19 period, logistical bottlenecks and container shortages drove up spice prices in both domestic and export markets. The government could mitigate such volatility through a spice-specific early-warning and response framework, combining satellite crop health monitoring with export order tracking and enabling rapid policy switches, like adjusting minimum export prices or extending freight subsidies to maintain domestic availability.


Urban-Rural Price Divergence
 Your CPI data show that urban retail prices for spices are consistently 5-15 index points higher than rural prices, due largely to branding, packaging costs and more fragmented retail chains in cities. In rural markets, prices tend to be lower but more volatile due to dependence on weekly haats and limited storage. Reducing these disparities requires government facilitation of direct procurement channels between rural producer groups and urban supermarkets, investment in spice grinding and packaging units at district level and the adoption of standardized grading norms to reduce variability in quality and price.


Contribution to Headline Inflation and Export Sensitivity
 While spices have a smaller weight in the CPI basket compared to staples like cereals or vegetables, their price surges can still contribute to food inflation, particularly when combined with simultaneous increases in other perishables. The export sensitivity of spices makes domestic prices vulnerable to international demand swings, as seen during periods when Indian turmeric and cumin shipments surged to the Middle East and Europe. For macroeconomic stability, the government should maintain a flexible export policy with built-in domestic availability triggers, expand buffer holdings for select spices with volatile prices, and integrate spice market intelligence into the Reserve Bank’s inflation monitoring to anticipate spillover effects.


Climate Change and Production Risk
 Increasing variability in rainfall patterns, higher average temperatures, and greater pest pressure are emerging risks for spice production. In recent years, erratic monsoon distribution has delayed sowing in Gujarat’s cumin belt and reduced turmeric yields in Telangana. The government should integrate spice crop CPI trends with IMD’s seasonal forecasts to anticipate shortages, promote heat- and drought-tolerant spice varieties through public breeding programs, and expand coverage of weather-based crop insurance tailored for high-value spices. Additionally, investments in solar-powered dehydration and storage technology can reduce post-harvest losses during prolonged wet periods, ensuring that price spikes from climate shocks are minimized.
 """,
        'Fuel and light': """ Long-Term Inflationary Trend
 The CPI for Fuel and Light has risen from about 106 in January 2013 to about 181 in July 2025, the latest month in the data (Rural+Urban), a cumulative increase of about 71 percent. This steady climb is largely structural, driven by global crude oil price trends, the depreciation of the rupee, periodic revisions in domestic fuel taxes and gradual increases in electricity tariffs. The impact of regulated pricing means that CPI movements in this category often reflect policy decisions as much as market forces. For the government, this underscores the importance of diversifying the national energy mix: accelerating investments in renewable energy under the National Solar Mission, expanding LNG import and distribution capacity to cushion crude oil shocks and implementing targeted subsidies for low-income households that ensure affordability without distorting market efficiency.


Seasonal Variation and Consumption Patterns
 While less seasonal than food categories, Fuel and Light shows minor cyclical changes linked to heating demand in winter months, festive season electricity usage, and pre-monsoon power surges. Rural areas, particularly in northern states, often record higher solid fuel prices in winter due to increased demand for firewood and biomass. Recognizing these patterns, the government could coordinate seasonal stockpiling of LPG cylinders in high-demand districts, incentivize efficient biomass stoves in rural households, and encourage time-of-day electricity pricing to manage peak load periods without triggering sharp CPI increases.


Price Shocks and External Disruptions
 The dataset reflects sudden spikes, such as in mid-2018 and late-2021, coinciding with global crude oil surges and revisions in excise duties. Other notable upticks align with currency depreciation episodes, which raise import costs for crude and LPG. In extreme cases, such as during early 2022 geopolitical tensions, wholesale and retail fuel prices in India rose sharply despite government excise cuts, highlighting supply chain and import dependency risks. The government can mitigate such shocks through a strategic petroleum reserve expansion, dynamic excise duty adjustment mechanisms tied to global benchmarks, and forward-contracting LNG imports to lock in stable prices.


Urban-Rural Divergence and Infrastructure Gaps
 Rural CPI for Fuel and Light often exceeds urban levels for kerosene, firewood, and other solid fuels due to limited supply points and higher transport costs, while urban consumers pay more for electricity and piped gas services. The dataset shows this divergence widening during logistics disruptions, such as monsoon-related road inaccessibility in rural areas. To close this gap, the government should expand last-mile LPG delivery infrastructure under the Pradhan Mantri Ujjwala Yojana, strengthen rural electricity distribution via Deen Dayal Upadhyaya Gram Jyoti Yojana, and promote decentralized renewable systems (solar micro-grids, biogas plants) to reduce dependency on transported fuels.


Contribution to Headline Inflation and Policy Signalling
 Fuel and Light carries significant weight in the CPI basket, meaning persistent increases can directly push headline inflation higher. For example, in mid-2021, rising petrol, diesel, and LPG prices contributed materially to headline CPI breaching the RBI’s 6 percent upper tolerance band, forcing monetary policy recalibration. Since these costs also feed into transport and production, they have indirect inflationary effects across the economy. For policy stability, the government could adopt a transparent fuel pricing formula that smooths volatility through variable taxes, coordinate fuel price interventions with monetary authorities, and release advance tariff change schedules for electricity to allow businesses and households to adjust.


Climate Change, Transition Risk, and Energy Security
 As India transitions towards a low-carbon economy, managing CPI stability during the shift is critical. The rollout of renewables, while reducing long-term fossil fuel exposure, must be balanced with short-term affordability. Extreme weather events, such as extended heatwaves, increase electricity demand and stress supply, driving tariff hikes. Meanwhile, global climate policies can influence fossil fuel availability and cost. To manage these transition risks, the government should integrate CPI trends into the National Energy Policy’s forecasting, accelerate grid-scale battery storage to stabilize renewable output and promote electrification of cooking and transport to reduce LPG and petrol dependency. Such measures would enhance both price stability and energy resilience.
 """,
        'Health': """ Long-Term Inflationary Trend
 The CPI for Health has shown a steady rise from about 104 in January 2013 to about 204 in July 2025, the latest month in the data (Rural+Urban), a cumulative increase of about 96 percent over the period. Unlike food and fuel, this trend is less influenced by short-term market volatility and more by structural cost drivers: rising prices of pharmaceuticals, higher consultation and diagnostic fees, increased cost of imported medical equipment and wage growth in the healthcare workforce. For the government, this underscores the need for systemic reforms to moderate healthcare inflation: strengthening the Pradhan Mantri Jan Arogya Yojana (PM-JAY) to expand free coverage, promoting domestic manufacturing of medical devices under the Production Linked Incentive (PLI) scheme and incentivizing generic drug adoption through Jan Aushadhi Kendras to lower retail medicine prices.


Seasonal Variation and Epidemic Cycles
 Health CPI shows mild seasonal patterns tied to disease outbreaks and climate-linked health events. Monsoon months often see increased demand for diagnostic services and medicines due to vector-borne diseases like dengue and malaria, leading to temporary price rises in pathology and treatment services. Winter months can see a similar uptick in respiratory illness-related spending. Policy measures should include advance procurement of essential medicines ahead of high-incidence seasons, mobile diagnostic units in high-risk districts, and fast-track insurance claim processing during epidemic months to avoid cost escalation from delayed care.


Price Shocks and External Disruptions
 The dataset captures noticeable upward jumps, such as during the COVID-19 pandemic, when the costs of PPE, oxygen supply, hospital stays, and diagnostics surged sharply. Similar localized shocks have been observed when imported raw materials for drugs (Active Pharmaceutical Ingredients from China) faced supply chain disruptions. The government can pre-empt such situations by diversifying API sourcing, building national reserves of critical medical consumables, and mandating transparent hospital pricing guidelines during declared public health emergencies to prevent opportunistic overcharging.


Urban-Rural Disparities and Accessibility Gaps
 Urban areas often exhibit higher CPI levels for healthcare due to greater reliance on private providers, higher real estate costs for hospitals, and premium pricing for branded drugs. Rural CPI for health is typically lower but masks hidden costs such as travel to distant facilities and delays in treatment. Addressing these gaps requires government investment in rural health infrastructure under the Ayushman Bharat Health and Wellness Centres initiative, telemedicine expansion to bridge specialist shortages, and subsidized transportation for patients requiring referral care in district hospitals.


Contribution to Headline Inflation and Social Stability
 Although the Health category carries a moderate weight in the CPI basket, its persistent inflation can have disproportionate social impact, as rising healthcare costs directly reduce disposable income and can push vulnerable households into debt. During periods of simultaneous food and healthcare inflation, the burden on low-income groups is particularly acute. For macroeconomic stability, the government could implement health cost monitoring alongside food inflation tracking, introduce price caps on essential medical services during crises, and expand preventive care programs that reduce high-cost interventions in the long run.


Climate Change and Emerging Health Risks
 The intersection of climate change and public health is increasingly visible in CPI movements, as climate-related disease burdens raise demand for medical services and pharmaceuticals. Heatwaves drive demand for hydration therapy and heat illness treatment; flooding increases vector-borne diseases; and pollution spikes increase respiratory care costs. To manage this, the government should integrate CPI health trends with climate and epidemiological forecasting, invest in public health resilience infrastructure (flood-proof hospitals, climate-controlled wards), and promote community-level preventive health programs. Additionally, supporting domestic vaccine R&D and manufacturing for climate-sensitive diseases could reduce import dependence and future price shocks. """,
    },
    'Regular User': {
        'Fruits': """ 1. Fruit Prices Have Been Rising for Years:
 Fruit prices have gone up by nearly 50 percent since 2013. This means buying the same amount of fruits today costs much more than it did ten years ago. For families, this is a reminder to look for cheaper seasonal fruits and buy in bulk when prices are lower. Storing fruits like apples or oranges in a fridge for longer use can help stretch your budget.
 
2. Prices Change With the Seasons:
 Certain fruits are cheap when they are in season and costlier when they are not. Mangoes, for example, are much cheaper in May and June than in March, while apples are most affordable around September and October. Knowing these cycles helps families plan their fruit shopping for the right months, saving money while enjoying fresher produce.
 
3. Sudden Price Spikes Can Happen:
 Sometimes fruit prices jump unexpectedly due to bad weather, transport problems, or reduced supply. This happened during the COVID-19 lockdown when many fruits became very expensive in cities. If prices suddenly rise, consider switching to other fruits or vegetables that are cheaper at the time until prices come down again.
 
4. City and Village Prices Can Be Different:
 Fruits can be much cheaper in rural areas where they are grown, while in cities prices are often higher due to transport and storage costs. If possible, buy directly from farmers' markets or weekly haats instead of supermarkets. This not only saves money but also supports local growers.
 
5. Fruits Affect the Cost of Living:
 When fruit prices go up, it can push up the cost of other things too, because fruits are part of the official cost-of-living measure. If you notice fruit prices climbing for several months, it’s a sign other daily essentials may also get costlier. Households can prepare by adjusting monthly budgets earlier rather than being caught off guard.
 
6. Weather and Climate Play a Big Role:
 Fruits depend heavily on the weather. Heatwaves, floods, or heavy rains can reduce supply and increase prices. If the news predicts poor weather for fruit-growing regions, expect prices to rise soon. Families can buy and store more durable fruits in advance or switch to alternatives that are less affected by weather problems. """,
        'Vegetables': """ Vegetable Prices Have Increased Over Time
 Since 2013, vegetable prices have gone up by around 55-60 percent. This means your monthly vegetable spending is higher now than a decade ago. Buying vegetables in local markets or directly from farmers can often be cheaper than supermarkets.


Prices Change With the Seasons
 Vegetables are cheapest during harvest time and more expensive in off-season months. Tomatoes, onions, and potatoes, for example, are cheaper just after harvest. Knowing these cycles can help families buy in bulk and store long-lasting vegetables like potatoes and onions for later use.


Sudden Spikes Can Happen
 Prices can rise sharply due to heavy rains, crop diseases, or transport strikes. When prices jump, try switching to alternative vegetables that are in season and cheaper at the time.


City and Village Prices Can Differ
 In villages, vegetables are often cheaper when they are grown locally, while in cities prices are higher due to storage and transport costs. Visiting weekly markets or buying directly from farmers can save money.


Vegetables Affect the Cost of Living
 Since vegetables are part of the cost-of-living index, a rise in vegetable prices can signal that other foods might get more expensive soon. If prices are rising steadily, plan your monthly budget in advance.


Weather Plays a Big Role
 Vegetable supply depends heavily on the weather. Floods, droughts, or unseasonal rains can lead to shortages and higher prices. If bad weather is expected, stock up on durable vegetables or frozen options before prices rise. """,
        'Spices': """ Spice Prices Have Been Rising for Years
 Since 2013, the cost of spices like turmeric, cumin, and coriander has increased by over 55 percent. This means everyday cooking is more expensive now than a decade ago. Buying in bulk during harvest season and storing in airtight containers can help save money.


Prices Change With the Harvest
 Spices are cheaper after harvest (for example, cumin and coriander prices drop between March and May), while prices rise in months with lower supply. Buying your annual stock soon after harvest can help you get better prices.


Unexpected Price Jumps Can Happen
 Poor weather, pests, or increased export demand can cause spice prices to rise suddenly. If this happens, try substituting with other spices or reducing usage in recipes until prices settle.


City and Village Price Gaps
 Urban markets often sell branded spices at higher prices, while in rural areas loose spices can be cheaper but may vary in quality. Families can save by buying from trusted wholesale markets and storing at home.


Spices Affect the Cost of Living
 Even though spices are a smaller part of the household budget, sharp increases can raise the overall cost of meals, especially when combined with rising vegetable or oil prices. If spice prices are going up, plan ahead for grocery costs.


Weather Impacts Spice Supply
 Heavy rains, drought, or heat can damage spice crops, leading to higher prices. If you hear about poor weather in spice-growing states, buy and store the spices you use most before prices rise.
 """,
        'Fuel and light': """ Fuel Prices Have Been Rising for Years
 Since 2013, the cost of fuel and electricity has gone up by more than 50 percent. This means filling your vehicle, using LPG gas, or paying electricity bills is more expensive now than a decade ago. Families can save by reducing unnecessary trips, using energy-efficient appliances, and cooking with pressure cookers to save gas.


Small Seasonal Changes Can Happen
 Fuel and electricity prices may go up slightly in certain months, like during winter in cold areas or during festivals when electricity use is high. Planning ahead by reducing usage in high-demand months can help keep bills under control.


Sudden Price Jumps Can Occur
 Global oil prices, currency changes, or government tax changes can quickly make fuel more expensive. When this happens, consider using public transport more often, carpooling, or combining errands into one trip to cut costs.


City and Village Price Differences
 In rural areas, firewood and kerosene can be more expensive due to transport costs, while in cities electricity and piped gas tend to cost more. Families should choose the most affordable option available in their area and avoid wasting fuel.


Fuel Costs Affect Everything Else
 When fuel prices rise, the cost of transporting goods also increases, which can make groceries and other daily needs more expensive. If fuel prices stay high for several months, expect other items to get costlier too, and adjust your budget accordingly.


Weather and Energy Use
 Heatwaves, cold spells, and heavy rains can all increase fuel and electricity use. Using solar water heaters, LED lights, and energy-saving fans can reduce bills during extreme weather and make households less dependent on expensive energy sources. """,
        'Health': """ Healthcare Costs Keep Climbing
 Since 2013, the cost of medical care has gone up by nearly 60 percent. This means doctor visits, medicines, and hospital treatments are much more expensive now than ten years ago. Families can save by using government health schemes like Ayushman Bharat or by buying medicines from Jan Aushadhi stores, which sell affordable generic drugs.


Seasonal Illnesses Can Affect Your Budget
 Medical expenses often rise during certain times of the year, like the monsoon (when diseases such as dengue and malaria become more common) or in winter (when respiratory illnesses increase). Families can prepare by keeping essential medicines and basic health supplies at home before these seasons start.


Unexpected Medical Price Jumps
 During emergencies, like the COVID-19 pandemic, the cost of medicines, oxygen, and hospital stays can rise sharply. Keeping basic first-aid items and common medicines at home can help avoid high prices during crises, and having health insurance can protect against sudden large bills.


Costs Vary Between Cities and Villages
 In cities, healthcare is often more expensive because people rely on private hospitals, while in villages costs are lower but facilities may be far away. If possible, families should use government hospitals for non-urgent care or preventive check-ups to reduce costs.


Health Costs Affect the Cost of Living
 As medical prices go up, they make life more expensive overall. If you notice medical bills getting higher month after month, adjust your family budget to set aside more for health expenses, even if you are not sick at the moment.


Weather and Climate Can Impact Health Spending
 Extreme heat, heavy rains or floods can lead to more illnesses, which can raise medical costs. If bad weather is predicted in your area, it’s smart to take preventive health steps (like boiling water, using mosquito nets or wearing masks) to reduce the risk of falling sick and spending more on treatment. """,
    }
}

    
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False
    if "analysis_audience" not in st.session_state:
        st.session_state.analysis_audience = None
    if "analysis_categories" not in st.session_state:
        st.session_state.analysis_categories = []

    btn1_col, btn2_col = st.columns([1, 1])
    show_one = btn1_col.button("Generate analysis for chosen audience & category")
    show_all = btn2_col.button("Generate analysis for ALL 5 categories (chosen audience)")

    if show_one:
        st.session_state.show_analysis = True
        st.session_state.analysis_audience = audience
        st.session_state.analysis_categories = [analysis_category]

    if show_all:
        st.session_state.show_analysis = True
        st.session_state.analysis_audience = audience
        st.session_state.analysis_categories = PRODUCT_LIST[:]

    
    if not st.session_state.show_analysis:
        st.info("Choose an audience and category above, then click a Generate button to see the analysis.")
    else:
        chosen_audience = st.session_state.analysis_audience
        chosen_categories = st.session_state.analysis_categories

        if not chosen_categories:
            st.info("No categories selected. Click one of the buttons above to generate analysis.")
        else:
            if len(chosen_categories) < len(PRODUCT_LIST):
                subtitle = ", ".join(chosen_categories)
            else:
                subtitle = "All 5 categories"
            lang_note = f" (in {language})" if language != "English" else ""
            st.subheader(f"Analysis for: {chosen_audience} • {subtitle}{lang_note}")

            for cat in chosen_categories:
                with st.expander(f"{chosen_audience} • {cat}", expanded=True):
                    original = ANALYSIS_TEXT.get(chosen_audience, {}).get(cat, "No analysis available.")
                    need_ai = (language != "English") or simplify

                    if not need_ai:
                        st.markdown(original)
                    else:
                        with st.spinner(f"Preparing the {cat} analysis in {language}..."):
                            result = ai_translate_simplify(original, language, simplify)

                        if result is None:
                            st.warning(
                                "AI translation/simplification needs the Gemini SDK and an API key, "
                                "so the original English text is shown below. See the setup notes at "
                                "the top of the script to enable it (pip install google-genai and add "
                                "GEMINI_API_KEY)."
                            )
                            st.markdown(original)
                        elif isinstance(result, str) and result.startswith("__ERROR__"):
                            st.error(
                                f"The AI request failed ({result[len('__ERROR__'):]}). "
                                "Showing the original English text instead."
                            )
                            st.markdown(original)
                        else:
                            st.markdown(result)
                        with st.expander("Show original (English)"):
                                st.markdown(original)



if page == "Inflation Forecasting":
    st.title("Forecasting Product CPI")
    col1, col2, col3, col4 = st.columns(4)

    forecast_type = col1.selectbox("How you want to see forecast", ["Specific date", "Over time"])
    product = col2.selectbox("Select product category", PRODUCT_LIST)
    sector = col3.selectbox("Select a sector", SECTORS)
    dt = col4.date_input("Forecast for date", format="YYYY-MM-DD")

    
    if dt is None:
        st.info("Please choose a date to forecast for.")
        st.stop()

    selected_month_start = pd.to_datetime(dt).replace(day=1)

    st.subheader(
        f'Forecasting for {sector} {product} CPI for the month of '
        f'{selected_month_start.strftime("%B %Y")}'
    )

    
    df = get_data()
    sector_df = df[df["Sector"] == sector].copy()
    sector_df.set_index("Month_Year", inplace=True)
    
    sector_df.index = sector_df.index.to_period("M").to_timestamp()
    sector_df.sort_index(inplace=True)
    
    sector_df = sector_df[~sector_df.index.duplicated(keep="first")]

    if product not in sector_df.columns:
        st.error(f"'{product}' is not a column in the dataset, so it cannot be looked up or forecast.")
        st.stop()

    
    data_start = sector_df.index.min()
    data_end = sector_df.index.max()

    if selected_month_start < data_start:
        
        st.warning(
            f'{selected_month_start.strftime("%B %Y")} is before the data begins '
            f'({data_start.strftime("%B %Y")}). Forecasting only works for dates after the '
            "available data, so no value can be shown for this month."
        )

    elif selected_month_start <= data_end:
        
        if selected_month_start in sector_df.index:
            value = sector_df.at[selected_month_start, product]
            if pd.isna(value):
                st.info(
                    f'{selected_month_start.strftime("%B %Y")} is within the data range, but there is '
                    f"no recorded {product} value for that month."
                )
            else:
                st.success(f'{selected_month_start.strftime("%B %Y")} is within the data range.')
                st.write(
                    f"The **{sector.lower()} {product.lower()} CPI** for the month of "
                    f'**{selected_month_start.strftime("%B %Y")}** is **{value:.2f}**.'
                )
        else:
            st.info(
                f'{selected_month_start.strftime("%B %Y")} falls inside the overall date range but '
                "is not present in the dataset (missing month), so no value can be shown."
            )

    else:
        
        st.warning(
            f'{selected_month_start.strftime("%B %Y")} is beyond the available data, so it will be forecast.'
        )

        month_diff = (selected_month_start.year - data_end.year) * 12 + (
            selected_month_start.month - data_end.month
        )

        if month_diff <= 0:
            st.error("Could not determine a valid forecast horizon for this date.")
        elif (product, sector) not in MODEL_PATHS:
            st.error(f"No forecasting model is available yet for {product} ({sector}).")
        else:
            model_fp = MODEL_PATHS[(product, sector)]
            try:
                model = update_to_data_end(load_model(model_fp), sector_df[product])
                forecasts = model.get_forecast(steps=month_diff)

                # Shared pieces used by both the output and the interpreter.
                _obs_series = sector_df[product]
                _fc_mean = forecasts.predicted_mean
                try:
                    _fc_ci = forecasts.conf_int()
                except Exception:
                    _fc_ci = None
                cpi_stats = compute_cpi_interpretation_stats(
                    _obs_series, _fc_mean, _fc_ci, product, sector
                )

                # Community-adjusted (blended) forecast: the model output is the
                # base; the community price trend nudges it within a capped band.
                # A category must have at least BLEND_MIN_RECEIPTS recent receipts
                # (capacity gate) before any trend is measured; otherwise the
                # forecast is shown exactly as the model produced it (unchanged).
                _adj = compute_community_adjustment(product)
                _factor = (1.0 + _adj["adj_pct"] / 100.0) if _adj["applied"] else 1.0
                _adj_mean = _fc_mean * _factor

                if forecast_type == "Specific date":
                    predicted = _fc_mean.iloc[-1]
                    st.write(
                        f"The forecasted **{sector.lower()} {product.lower()} CPI** for the month of "
                        f'**{selected_month_start.strftime("%B %Y")}** is **{predicted:.2f}**.'
                    )
                    if _adj["applied"]:
                        _adj_val = _adj_mean.iloc[-1]
                        _mc, _ac = st.columns(2)
                        _mc.metric("Model forecast", f"{predicted:.2f}")
                        _ac.metric(
                            "Community-adjusted",
                            f"{_adj_val:.2f}",
                            f"{_adj['adj_pct']:+.1f}%",
                        )
                else:
                    future_mean = _fc_mean
                    future_ci = _fc_ci if _fc_ci is not None else forecasts.conf_int()

                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(sector_df.index, sector_df[product], label="Observed")
                    plt.plot(future_mean.index, future_mean, label="Forecast", color="green")
                    plt.fill_between(
                        future_ci.index,
                        future_ci.iloc[:, 0],
                        future_ci.iloc[:, 1],
                        color="green",
                        alpha=0.2,
                    )
                    if _adj["applied"]:
                        plt.plot(
                            _adj_mean.index,
                            _adj_mean,
                            label="Community-adjusted",
                            color="darkorange",
                            linestyle="--",
                            linewidth=2,
                        )
                    plt.title(f"{sector} {product} CPI Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("CPI")
                    plt.grid(True)
                    plt.legend(loc="upper left")
                    st.pyplot(fig)
                    plt.close(fig)

                # Explain the community adjustment, or show progress toward the
                # capacity threshold when a category doesn't yet have enough data.
                if _adj["applied"]:
                    _dir_word = "higher" if _adj["adj_pct"] > 0 else "lower"
                    _trend = "rising" if _adj["adj_pct"] > 0 else "falling"
                    st.caption(
                        f"**Community-adjusted forecast:** recent community receipts for "
                        f"{product.lower()} show prices **{_trend}**, so the model forecast is nudged "
                        f"**{abs(_adj['adj_pct']):.1f}% {_dir_word}** (from {_adj['n_recent']} recent "
                        f"receipt item(s), weighted for confidence and capped at ±{BLEND_MAX_ADJ_PCT:.0f}%). "
                        "The model's own forecast is shown unchanged alongside it. The adjusted figure is a "
                        "real-world overlay you can weigh yourself, not a change to the underlying model."
                    )
                elif _adj.get("reason") == "too_few_recent" and _adj.get("n_recent", 0) >= 1:
                    _need = _adj.get("min_receipts", BLEND_MIN_RECEIPTS)
                    st.caption(
                        f"📊 *Building community data for {product.lower()}:* "
                        f"**{_adj['n_recent']}/{_need}** recent receipts collected. Once this category "
                        f"reaches {_need} recent receipts, a community-adjusted trend will appear here "
                        "automatically. Until then, the forecast above is the model's own output."
                    )

                # CPI interpreter: dynamic per graph/value, works offline.
                _ctx = f"{product}|{sector}|{selected_month_start:%Y-%m}|{forecast_type}"
                _btn = (
                    "Interpret this graph: what does this CPI mean?"
                    if forecast_type == "Over time"
                    else "Know more: what does this CPI value mean?"
                )
                render_cpi_interpretation(cpi_stats, context_key=_ctx, button_label=_btn)

                # Community price signal from the shared receipts sheet: a
                # real-world cross-check that can inform how you read the forecast.
                render_crowd_signal(
                    product,
                    model_direction=cpi_stats.get("direction") if isinstance(cpi_stats, dict) else None,
                    context_key=_ctx,
                )

            except FileNotFoundError:
                st.error(
                    f"The model file could not be found at:\n{model_fp}\n\n"
                    "Check MODELS_DIR and the file name in the CONFIG block."
                )
            except Exception as e:
                st.error(f"Model not ready yet: {e}")



if page == "Crowdsourcing Receipts":
    st.title("Crowdsource Your Receipts")
    st.write(
        "Help build a real-time, community-driven picture of local prices by sharing your receipts. "
        "Upload a clear photo of a paper receipt (PNG/JPG) or a PDF invoice, and the app uses AI to "
        "read the merchant, date, total, and line items automatically."
    )

    # --- Privacy consent gate (GDPR/DPDP-friendly): require informed opt-in
    #     before any receipt is uploaded or sent to the AI service. ---
    if "privacy_consent" not in st.session_state:
        st.session_state.privacy_consent = False

    st.info(
        "**Before you upload, please read.** Receipts can contain personal "
        "information (your name, card details, location or other people's "
        "details). Ayara processes each file **only for this session** to extract "
        "prices and does not store your receipts afterwards. To read them "
        "automatically, the file is sent to Google's Gemini AI service for "
        "processing. Full details are in **Privacy & Terms** in the menu."
    )
    _consent = st.checkbox(
        "I have read the Privacy Policy and Terms. I am at least 16 years old, I "
        "have the right to share these receipts, and I have removed or obscured "
        "any personal details (mine or others') that I do not want processed.",
        key="privacy_consent_checkbox",
    )
    st.session_state.privacy_consent = _consent

    if not st.session_state.privacy_consent:
        st.warning(
            "Please confirm the statement above to enable receipt uploading and "
            "reading. You can review the full terms under **Privacy & Terms**."
        )
        st.stop()

    if not ai_ready():
        if not GEMINI_AVAILABLE:
            st.warning(
                "Automatic reading needs the Gemini SDK. Install it with "
                "`pip install google-genai` and restart the app. You can still upload files below."
            )
        else:
            st.warning(
                "Automatic reading needs a Gemini API key. Add GEMINI_API_KEY to "
                "`.streamlit/secrets.toml` (see the setup note at the top of the script). "
                "You can still upload files below; they just won't be read automatically yet."
            )

    _used, _cap = quota_status()
    st.caption(f"Free-tier usage today: {_used}/{_cap} AI requests (auto-limited to protect your quota).")

    uploaded = st.file_uploader(
        "Upload receipt(s): photo, screenshot or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("No receipts uploaded yet. Choose one or more files above to contribute.")
    else:
        if len(uploaded) > MAX_FILES_PER_RUN:
            st.warning(
                f"You uploaded {len(uploaded)} files; only the first {MAX_FILES_PER_RUN} "
                "will be processed at once to stay inside the free tier."
            )

        all_items = []
        for f in uploaded[:MAX_FILES_PER_RUN]:
            st.divider()
            st.subheader(f"Receipt: {f.name}")

            raw_bytes = f.getvalue()
            size_mb = len(raw_bytes) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(
                    f"This file is {size_mb:.1f} MB, over the {MAX_UPLOAD_MB} MB limit. "
                    "Please upload a smaller or more compressed file."
                )
                continue

            is_pdf = f.name.lower().endswith(".pdf") or (f.type == "application/pdf")

            
            if is_pdf:
                send_bytes, mime = raw_bytes, "application/pdf"
                st.caption("PDF detected, sending to the reader.")
            elif PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(raw_bytes))
                    st.image(img, caption=f.name, width=320)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    send_bytes, mime = buf.getvalue(), "image/jpeg"
                except Exception:
                    st.error("That image could not be opened. Please try a different file.")
                    continue
            else:
                send_bytes, mime = raw_bytes, (f.type or "image/jpeg")

            if not ai_ready():
                st.info("File received. Add a Gemini API key to read it automatically.")
                continue

            with st.spinner("Reading the receipt with AI..."):
                data = ocr_receipt(send_bytes, mime)

            if "__error__" in data:
                st.error(data["__error__"])
                continue

            
            c1, c2, c3 = st.columns(3)
            c1.metric("Merchant", str(data.get("merchant_name") or "-"))
            c2.metric("Date", str(data.get("transaction_date") or "-"))
            total = data.get("total")
            currency = data.get("currency") or ""
            c3.metric("Total", f"{currency} {total}".strip() if total is not None else "-")

            
            items = data.get("items") or []
            if items:
                try:
                    items_df = pd.DataFrame(items)
                    items_df.insert(0, "merchant", data.get("merchant_name"))
                    items_df.insert(1, "date", data.get("transaction_date"))
                    items_df["currency"] = data.get("currency")
                    st.dataframe(items_df, use_container_width=True)
                    all_items.append(items_df)
                except Exception:
                    st.json(data)
            else:
                st.caption("No line items were detected on this receipt.")
                st.json(data)

        
        if all_items:
            combined = pd.concat(all_items, ignore_index=True)
            st.divider()
            st.subheader("All parsed items")
            st.dataframe(combined, use_container_width=True)
            st.download_button(
                "Download parsed items as CSV",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="ayara_parsed_receipts.csv",
                mime="text/csv",
            )
            st.caption(
                "These results are kept only for this session unless you choose to "
                "contribute them to the shared dataset below."
            )

            # ---- Contribute to the shared price dataset (Google Sheets) ----
            st.divider()
            st.subheader("Contribute to the shared price dataset")

            # Categorise every parsed item locally (free, offline) for preview + saving.
            _preview = combined.copy()
            for _c in ["date", "description", "quantity", "unit_price",
                       "line_total", "currency", "merchant"]:
                if _c not in _preview.columns:
                    _preview[_c] = ""
            _preview["category"] = _preview["description"].apply(categorize_item)

            _summary = (
                _preview["category"]
                .value_counts()
                .reindex(PRODUCT_LIST + ["Other"], fill_value=0)
            )
            st.caption(
                "Each item is sorted into the same categories as your inflation graphs; "
                "anything that doesn't match goes to **Other**."
            )
            _scols = st.columns(len(PRODUCT_LIST) + 1)
            for _i, _cat in enumerate(PRODUCT_LIST + ["Other"]):
                _scols[_i].metric(_cat, int(_summary[_cat]))

            with st.expander("Preview how items were categorised", expanded=False):
                st.dataframe(
                    _preview[["date", "category", "description", "quantity",
                              "unit_price", "line_total", "currency", "merchant"]],
                    use_container_width=True,
                )

            _keep_other = st.checkbox(
                "Also store items that didn't match a category (under 'Other'). "
                "Uncheck to discard them.",
                value=True,
                key="keep_other_items",
            )
            st.caption(
                "ℹ️ Saving adds this categorised price data (item, price, date, store) to the shared "
                "community Google Sheet so it can inform the inflation picture. Please don't save "
                "receipts containing personal details you wouldn't want shared."
            )

            if not GSPREAD_AVAILABLE:
                st.warning(
                    "Saving to the shared sheet needs the `gspread` package. Install it with "
                    "`pip install gspread google-auth` and restart the app."
                )
            elif not sheets_ready():
                st.warning(
                    "Google Sheets isn't connected yet. Add a service-account key "
                    "(in Streamlit secrets as `[gcp_service_account]`, or a local "
                    "`ayara_service_account.json` in your Resources folder), **share the sheet "
                    "with that service account's email**, then restart the app."
                )
            else:
                if st.button("💾 Save to shared Google Sheet", type="primary"):
                    with st.spinner("Saving to the shared sheet..."):
                        try:
                            _res = save_receipts_to_sheet(combined, keep_other=_keep_other)
                        except Exception as e:
                            _res = None
                            st.error(f"Could not save to the sheet: {e}")
                    if _res is not None:
                        if _res["appended"]:
                            _by_cat = ", ".join(
                                f"{k}: {v}" for k, v in _res["counts"].items() if v
                            )
                            st.success(
                                f"Saved {_res['appended']} new item(s) to the shared sheet"
                                + (f" ({_by_cat})." if _by_cat else ".")
                            )
                        else:
                            st.info("Nothing new to add: these items were already in the sheet.")
                        _notes = []
                        if _res["skipped_dupes"]:
                            _notes.append(f"{_res['skipped_dupes']} duplicate(s) skipped")
                        if _res["skipped_other"]:
                            _notes.append(f"{_res['skipped_other']} uncategorised item(s) discarded")
                        if _notes:
                            st.caption(" · ".join(_notes))
                        st.caption(
                            "Newest entries are sorted to the top of the sheet automatically, and "
                            "the data feeds the **Community price signal** on the Forecasting page."
                        )

if page == "Privacy & Terms":
    render_privacy_and_terms()
