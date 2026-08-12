# Piston Kinetic Theory Simulator

A gas of N ~ 10^23 particles heated in a piston (free/isobaric or
locked/isochoric), split into:

- **`main.py`** — FastAPI backend. Owns the authoritative thermodynamic
  state and steps it forward via the ideal gas law and first law of
  thermodynamics (`main.py` docstring has the details). Exact N is used
  for all real physics.
- **`static/index.html`** — Frontend. Renders a representative sample of
  ~180 tracer particles for visual intuition and polls the backend for
  T, P, V, U, W, and the P–V trace. It does **not** compute the
  thermodynamics itself.

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** — FastAPI serves the frontend
directly, so there's no separate dev server or CORS setup needed.

## API

| Endpoint      | Method | Body                                   | Returns               |
|---------------|--------|-----------------------------------------|------------------------|
| `/api/reset`  | POST   | `SimParams` (N, gas type, V0, T0, mode, heat, duration) | fresh `SimState` |
| `/api/step`   | POST   | `{"dt": 0.15}`                          | updated `SimState`    |
| `/api/state`  | GET    | —                                        | current `SimState`    |

`SimParams` and `SimState` are Pydantic models defined in `main.py` —
visiting `http://127.0.0.1:8000/docs` gives you interactive Swagger docs
with full schemas and a try-it-out console.

## Extending

- Swap the piston's external-pressure assumption (currently pinned to the
  initial equilibrium P₀) for a real atmospheric-pressure + piston-mass
  parameter.
- Add a `/api/maxwell` endpoint returning an actual Maxwell–Boltzmann
  speed histogram at the current T, rather than the visual-only
  √(T/T₀) scaling used for tracer particles now.
- Persist state per session (e.g. keyed by a client id) if you want more
  than one simulation running concurrently — right now `_Sim` is a single
  global instance, fine for local single-user use.
