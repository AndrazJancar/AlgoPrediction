PY=python

.PHONY: install predict train backtest sync docs health

install:
	$(PY) -m pip install --upgrade pip
	pip install -r requirements.txt

predict:
	$(PY) -m src.predict

train:
	$(PY) -m src.train

backtest:
	$(PY) -m src.backtest

sync:
	$(PY) scripts/sync_forecasts_to_docs.py

docs:
	$(PY) scripts/daily_run.py

health:
	$(PY) scripts/health_check.py


