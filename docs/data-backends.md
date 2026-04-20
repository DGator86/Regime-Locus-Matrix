# Data backends

RLM runtime now supports `--backend auto|csv|lake`.

* `auto`: prefer lake data and fall back to CSV.
* `csv`: force classic `data/raw/*.csv` files.
* `lake`: force parquet lake files under `data/lake/<SYMBOL>/`.
