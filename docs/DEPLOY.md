# Ayara: CPI & Inflation Insights

Ayara is a Streamlit app for exploring India's Consumer Price Index (CPI): it
shows inflation analysis by category, produces CPI forecasts from pre-trained
time-series models, reads prices off uploaded receipts and folds those
community-submitted prices back into the forecast as a capped, real-world
adjustment.

## Repository layout

Streamlit Community Cloud runs `streamlit run` from the repository root. So the
app's `Resources` folder must live **inside the repo** at the root (the app
looks for `./Resources` when the usual `~/Documents/Resources` path isn't
present, e.g. on the cloud):

```
your-repo/
├── ayara_app.py                # the app (entrypoint)
├── requirements.txt            # Python dependencies
├── .gitignore
├── .streamlit/
│   ├── config.toml             # theme (safe to commit)
│   └── secrets.toml            # secrets, NOT committed (see below)
└── Resources/
    ├── Ayara.png
    ├── Inflation.jpg
    ├── Revised Final Dataset.csv
    └── Time Series Models/
        ├── vegetables-urban.pkl.xz
        ├── vegetables-rural.pkl.xz
        ├── Vegetables-Total.pkl.xz
        ├── Fuel_and_light-urban.pkl.xz
        ├── Fuel_and_light-rural.pkl.xz
        ├── Fuel_and_light-total.pkl.xz
        ├── Health-urban.pkl.xz
        ├── Health-rural.pkl.xz
        ├── Health-Total.pkl.xz
        ├── Fruits-urban.pkl.xz
        ├── Fruits-Total.pkl.xz
        ├── Spices-Urban.pkl.xz
        ├── spices-rural.pkl.xz
        ├── spices-total.pkl.xz
        ├── Fruits-Rural.pkl.xz.part001   # Fruits-Rural is split into 2 parts
        └── Fruits-Rural.pkl.xz.part002   # commit BOTH
```

The models are stored compressed (`.pkl.xz` or split `.pkl.xz.partNNN` chunks)
so every file stays under GitHub's 25 MB browser-upload limit. The app
decompresses them in memory at load time; the decompressed bytes are identical
to the original `.pkl` files. So forecasts are unchanged. You do **not** need to
commit the large raw `.pkl` files (they're git-ignored).

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (including the `Resources/` folder and the
   compressed model files).
2. On https://share.streamlit.io create a new app pointing at your repo, branch
   and `ayara_app.py`.
3. In **Advanced settings**, choose **Python 3.12** (or 3.11). This is required:
   the models need `numpy` 1.x, which has no Python 3.13 wheels.
4. Add your secrets in **Settings → Secrets** (see below). Deploy.

## Secrets

Secrets are never committed. Locally they live in `.streamlit/secrets.toml`; on
the cloud you paste the same content into **Settings → Secrets**. The app uses:

```toml
# Google Gemini (receipt OCR + AI features)
GEMINI_API_KEY = "your-key"

# Google service account (writing/reading the shared price Google Sheet)
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

Also remember to **share the Google Sheet** with the service account's
`client_email` (Editor access) or writes/reads will be refused.

## Running locally

```bash
pip install -r requirements.txt
streamlit run ayara_app.py
```

Run it from the repository root so `./Resources` resolves correctly. Locally you
can keep either the compressed models or the original `.pkl` files in
`Resources/Time Series Models/`. The app checks for a plain `.pkl` first, then
the compressed forms.

## Notes

- `requirements.txt` pins `numpy`, `scipy`, `pandas` and `statsmodels` for
  compatibility with the pickled models. If a model fails to load, match those
  four to the versions from the machine where you trained them.
- AI (Gemini) and Google Sheets features degrade gracefully: if their keys
  aren't configured, the rest of the app still runs.
