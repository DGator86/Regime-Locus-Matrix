# tools/diagnostics/

One-off diagnostic and inspection utilities.

For the primary health-check entrypoint use:

```bash
rlm doctor
```

Scripts here provide deeper / provider-specific diagnostics:

| Script | Description |
|--------|-------------|
| `diagnose_ibkr.py` | IBKR TWS connection + account diagnostics |
| `diagnose_massive_flatfiles_s3.py` | Massive S3 flat-file connectivity check |
| `super_ping_data.py` | Multi-provider data quality ping |
