# Staging deployment - Streamlit Community Cloud (free)

For client testing. One shareable link, viewing can be limited to the client's email.

> The repo can stay public: the price workbook is committed only **encrypted**
> (`price_lists/HCC.xlsm.enc`). The app unlocks it in memory with the `PRICE_SHEET_KEY` secret.

## How it runs on Streamlit Cloud

- One process: the chat app. There is no Redis / Celery worker there, so uploaded PDFs and images are
  read in a **background thread inside the app** (the chat stays usable, progress card still shows).
  This is automatic - nothing to configure.
- Storage (chats, customer memory, quotes, uploads) lives on the app's disk and is **reset when the
  app restarts or is redeployed**. Fine for staging; production needs a server or a database.
- The free app sleeps after a few days without visitors; opening the link wakes it in ~30 seconds.

## Steps

1. Deploy branch: **`staging`** (push changes there to update the staging app).
2. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns the repo
   (`wp113department-cell`).
3. **Create app** -> *Deploy a public app from GitHub* is fine even for a private repo:
   - Repository: `wp113department-cell/Data_analysis`
   - Branch: `staging`
   - Main file path: `main.py`
   - Advanced settings -> Python version: **3.12**
4. **Advanced settings -> Secrets** - paste this and fill in the values (never commit them):

   ```toml
   GROQ_API_KEY = "..."
   OPENROUTER_API_KEY = "..."
   GEMINI_API_KEY = "..."
   CLOUDFLARE_API_TOKEN = "..."
   CLOUDFLARE_ACCOUNT_ID = "..."
   PRICE_SHEET_KEY = "..."           # unlocks the encrypted price workbook
   STAFF_PIN = "choose-a-pin"        # optional: unlocks Staff mode (teach rules, see "Needs you")
   ```

   Streamlit Cloud exposes these top-level secrets as environment variables, which is what the app reads.
5. **Deploy**. The first build takes a few minutes (it installs `requirements.txt`).
6. **Share with the client:** app menu (top right) -> *Share* -> make the app **private** and add the
   client's email addresses. They sign in with that email to open it.

## After deploying - quick checks

- Sidebar shows **HCC** under *Price sheets (live)* with a green dot (the encrypted workbook was unlocked).
- Send "Merry Christmas!" (fast reply) and "quote 2 W2136 in white shaker, not assembled, pickup".
- Attach a layout image - the progress card appears and the agent continues on its own.

## Updating the staging app

Push to the `staging` branch - Streamlit Cloud redeploys automatically. To change prices, update the
workbook locally, re-run `python scripts/encrypt_price_sheet.py "Copy of HCC Signature Series Quote.xlsm"
price_lists/HCC.xlsm.enc` (same key from `.env`), commit the `.enc` file and push.
