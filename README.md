# AlgoPrediction
Algoritem, ki dela predikcije za en dan vnaprej (BSP/Borzen).

## GitHub Pages (napovedi)

Ta repo vsebuje statično stran v `docs/` za ogled napovedi. Deploy poteka prek GitHub Actions workflowa v `.github/workflows/pages.yml`.

- URL po uspešnem deployu: `https://<username>.github.io/<repo>/`
- Viewer bere datoteke iz `docs/data/manifest.json` (najnovejša prva).

### Avtomatska objava
Ob vsakem pushu na `main` se zažene:
1. `python scripts/sync_forecasts_to_docs.py` — kopira `out/forecast_*.json` v `docs/data/` in zgradi `manifest.json`.
2. Objavi mapo `docs/` na GitHub Pages.

### Ročno osveževanje lokalno
```bash
python scripts/sync_forecasts_to_docs.py
```
Commit + push na `main` sproži deploy.

## Daily avtomatika

Workflow `.github/workflows/daily.yml` vsak dan:
- zažene `scripts/daily_run.py` (predikcija, snapshoti Borzen/BSP, sync v `docs/`)
- objavi `docs/` na GitHub Pages

Potrebne skrivnosti (Settings → Secrets and variables → Actions):
- `ENTSOE_API_KEY`: ključ za ENTSO-E API

Urejanje urnika: v `daily.yml` spremenite `cron: '15 17 * * *'` (UTC).
