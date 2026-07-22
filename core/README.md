# polarspike-core

The minimal core of [polarspike](https://github.com/marvinseifert/polarspike): load
MEA recordings, query spikes against stimulus triggers, and work with the resulting
dataframes. No plotting, no clustering, no GUI.

```bash
pip install polarspike-core
```

```python
from polarspike import Overview

rec = Overview.Recording.load("my_recording.pkl")
spikes = rec.get_spikes_triggered([[0]], [["all"]], pandas=False)
```

Installs six dependencies (pandas, numpy, polars, pyarrow, h5py, scipy).

`Recording.show_df()` returns a plain `pandas.DataFrame` here. For the interactive
Panel table, install `polarspike-core[interactive]`, or the full `polarspike`
distribution for the analysis and plotting modules.