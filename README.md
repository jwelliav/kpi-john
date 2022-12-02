# kpi

Simplified / stripped-down version of Kyohei-san's (or Yoshi-san's?) KPI metrics from `proposal-evaluate`.

## Install

Clone and then install with
```shell
python -m pip install -e kpi
```

### Docs

If you also want to build the docs locally, install with
```shell
python -m pip install -e kpi[docs]
```

Or, if you are using `zsh` specifically, with
```shell
python -m pip install -e kpi\[docs\]
```

## Usage

```python
from kpi import get_metrics_from_y_values

metrics = get_metrics_from_y_values(df)
```
