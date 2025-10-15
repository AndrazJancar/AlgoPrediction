# Podatkovni viri za napovedovanje energijskih cen

## Trenutno implementirani viri

### ENTSO-E Transparency Platform
- **API:** `entsoe-py` knjižnica
- **Podatki:** Day-ahead cene, load forecasts, generation forecasts
- **Frekvenca:** 15-minutni intervalli (resampled iz urnih)
- **Endpoint:** `https://web-api.tp.entsoe.eu/api`

## Potrebni dodatni viri

### 1. Borzen.si - Slovenija
- **URL:** https://borzen.si/sl-si/
- **Podatki:**
  - Imbalance prices (neuravnoteženost)
  - Market clearing prices
  - Bilančni obračuni
  - Platforma za izravnalno energijo
- **Implementacija:** Potrebno dodati web scraping ali API dostop
- **Pomembnost:** Visoka - direktni podatki o slovenskem trgu

### 2. BSP South Pool
- **URL:** https://www.bsp-southpool.com/domov.html
- **Podatki:**
  - Day-ahead market results
  - Intraday market data
  - Cross-border capacities (CZC)
  - Market coupling data
- **Implementacija:** Potrebno dodati web scraping ali API dostop
- **Pomembnost:** Visoka - regionalni trgni podatki

## Predlagana implementacija

### Web Scraping (Python)
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_borzen_data():
    # Implementacija za Borzen.si
    pass

def scrape_bsp_data():
    # Implementacija za BSP South Pool
    pass
```

### API dostop (če je na voljo)
- Preveriti, ali imata Borzen ali BSP javne API-je
- Implementirati direktni dostop do podatkov

## Prednosti dodatnih virov

1. **Boljše napovedi** - več podatkov = boljša natančnost
2. **Regionalni kontekst** - razumevanje slovenskega in regionalnega trga
3. **Imbalance pricing** - pomembno za risk management
4. **Market coupling** - vpliv sosednjih trgov na cene

## Naslednji koraki

1. Analizirati strukturo podatkov na obeh spletnih mestih
2. Implementirati web scraping funkcije
3. Integrirati podatke v ETL pipeline
4. Dodati nove features v model
5. Testirati izboljšanja natančnosti
