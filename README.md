# mortgage-calculator

Python calculator which can be used to compare multiple mortgage loan estimates
simultaneously.

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
mortgage example.json  # May take a few seconds
```

### Windows

```shell
python3 -m venv venv
.\venv\Scripts\activate
mortgage.exe example.json  # May take a few seconds.
```

Then create a new JSON file with the loan estimates you have from lenders and run the
calculator again.
