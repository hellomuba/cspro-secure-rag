# AegisRAG Security Console (optional UI)

A thin Streamlit frontend over the existing HTTP API. It is **not** part of the
assessment deliverable and nothing in `app/` depends on it — delete this
directory and the service behaves identically.

## The one rule this UI follows

**It is a client, not a component of the system.** It contains no business
logic, no tenant logic and no security logic. It sends HTTP requests with an
API key and renders what comes back.

That means:

- **Every verdict on screen comes from a live API response.** No panel is
  hardcoded to say "secure". The security tab evaluates the *actual* response
  body — if the API ever leaked the canary, the panel would go red.
- **Choosing a tenant chooses which API key to send**, nothing more. The UI
  cannot grant itself access; the server resolves the tenant from the key, as
  it does for any other client.
- **The UI cannot weaken the backend.** It has no privileged endpoint, no
  bypass, and no ability to disable a control.

## Run it

The API must already be running.

```powershell
# 1. install (one time)
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt

# 2. start the API in one terminal
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8001

# 3. start the console in another
.\.venv\Scripts\streamlit.exe run ui/app.py
```

Opens on <http://127.0.0.1:8501>.

`.streamlit/config.toml` in the repository root pins three defaults that
matter: the server binds **loopback only** (Streamlit's own default binds every
interface, which would expose a console holding tenant API keys to the LAN),
telemetry is off, and tracebacks are not rendered into the browser.

Point it at a different API with `AEGISRAG_API_URL`, and pre-fill the keys with
`TENANT_A_API_KEY` / `TENANT_B_API_KEY`:

```powershell
$env:AEGISRAG_API_URL = "http://127.0.0.1:8001"
$env:TENANT_A_API_KEY = "dev-a-local-demo-key-001"
$env:TENANT_B_API_KEY = "dev-b-local-demo-key-002"
.\.venv\Scripts\streamlit.exe run ui/app.py
```

Otherwise type the keys into the sidebar; they are password fields and are held
only in Streamlit session state.

## What it shows

**System health** — provider, model, whether inference is `local` or `remote`,
whether the runtime was actually reached, and per-tenant record counts. A
remote provider gets a warning banner, because a cloud call must never be
presented as local inference.

**RAG playground** — ask as either tenant, with `top_k`. Renders the answer,
the verified citations with excerpts, any superseded records that were excluded
and why, the security flags, and the audit id. A citation belonging to another
tenant would be rendered in red as a leak; in normal operation that is
unreachable.

**Security demonstrations** — four scenarios run against the live API:
cross-tenant canary exfiltration, a forged `[SYSTEM]` block in the question,
an enumerate-every-tenant request, and the Tenant B policy conflict. Each
response is checked for the canary anywhere in the body, for foreign-tenant
citations, and for foreign record ids, with PASS/FAIL from the real result.

**Ingestion** — reproduces the incident: import Tenant B's forged-SYSTEM
document into Tenant A's namespace and watch it get quarantined, then repeat
the import and watch the retry come back `unchanged`.

## Why Streamlit rather than a React/TypeScript SPA

Not a matter of taste. A browser-side SPA would have forced two changes that
work against the point of this project:

1. **CORS middleware on the API.** A React app served from another origin needs
   it. Adding permissive cross-origin configuration to a security-demonstration
   service, for cosmetics, is a poor trade.
2. **Tenant API keys in browser JavaScript.** The core claim here is that
   tenant identity comes from a credential the client cannot forge. A key
   shipped into browser JS is visible in devtools and the network tab. Doing it
   properly would mean a backend-for-frontend proxy holding the keys
   server-side — more code and more attack surface for a console.

Streamlit avoids both: it executes server-side, so requests to the API are
issued by the Streamlit process, not the browser. No CORS entry is needed and
no key is used in browser JavaScript.

## Limitations

- **No authentication on the console itself.** It is a local developer tool.
  Do not expose it; anyone who can reach it can use whichever keys are loaded.
- **Keys are typed into the browser and held in server-side session state.**
  They are not persisted and are not used from browser JavaScript, but they do
  travel from the browser to the Streamlit process. Use the synthetic
  development keys only.
- **Local inference takes roughly 20 seconds per question on CPU.** The spinner
  is not stuck.
- The console reflects whatever the API returns. It cannot detect a backend
  that is lying to it — it is a demonstration aid, not an audit tool.
