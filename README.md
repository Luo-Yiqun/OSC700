# OSC700

A toolkit for building a dataset of organic semiconductor crystals, combining many-body excited-state calculations (GW/BSE with Quantum ESPRESSO and BerkeleyGW), automated literature review of experimental and computational references, and feature data for machine-learning workflows. The current version is preliminary and under active development; the calculation data itself is not included in this repository.

## Repository layout

- **`codes/`** — Utility scripts for the calculation workflow: CIF → JSON conversion (`writeJSON.py`, `writeJSON_batch.py`), spectrum plotting (`DrawSpectrum.py`), histograms (`hist.py`), and structure I/O helpers adapted from [dbaAutomator](https://github.com/Xingyu-Alfred-Liu/dbaAutomator).
- **`LiteratureReview/`** — Automated literature-review pipeline (`main.py`, `reviewer.py`, `citer.py`): searches Google Scholar for papers citing each crystal's source publication, screens them by keyword and with an LLM reviewer for experimental/computational optical-property data. Curated results are stored in the `pah101*`, `kaiji*`, `osc_tadf*`, and energy-binned (`100-125ev` etc.) text/JSON files.

## Setup

```bash
pip install -r requirements.txt
```

For the literature-review pipeline you additionally need:

1. **An OpenAI API key** — put your own key in `LiteratureReview/config.json` (the file in this repo contains a placeholder):

   ```json
   {
       "openai_api_key": "sk-..."
   }
   ```

2. **Google Chrome** — the Scholar search uses Selenium, which manages a matching ChromeDriver automatically (Selenium ≥ 4.6); you only need a local Chrome installation.

The CSD Python API (`ccdc`) is optional and proprietary; scripts import it under `try/except`.
