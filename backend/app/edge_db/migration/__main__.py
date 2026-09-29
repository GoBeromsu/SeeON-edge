"""Run the cutover commands as ``python -m backend.app.edge_db.migration``."""

from backend.app.edge_db.migration.cli import main

raise SystemExit(main())
