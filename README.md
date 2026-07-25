# PyFluent MCP (core, no-KB)

**Developed by Vijaya Sankaran K**

**License:** [PolyForm Noncommercial 1.0.0](LICENSE) — free for noncommercial use; commercial use requires a separate license (see [License](#license) below).

**Bare Fluent tools server — no CFD Intelligence / knowledge-base layer.**
This is the `pyfluent-mcp-v5` package with the classification/phase-plan/
KB-retrieval layer (`kb_router.py`, `kb_server.py`, `knowledge/`, and the
static registry/menu/SOP JSON files) removed. Everything that talks to
Fluent directly — connect/launch, meshing, boundary conditions, solver
setup, run control, post-processing, exports — is unchanged and lives
entirely in `pyfluent_mcp_server.py`, which is fully self-contained and
runnable on its own.

## What's included (tools layer v3.11.0 — 2026-07-20)

Hardened across three live manifold sessions (geometry → mesh → converged
solve → post-processing), fixing every incident class encountered:

- **Resilient sessions** — endpoint + context persisted to `logs/last_session.json`;
  `check_fluent_connection` auto-reattaches after an MCP restart;
  `reattach_last_session` for explicit recovery. Busy Fluent is reported as
  `BUSY_OR_UNRESPONSIVE`, never "dead". Full Fluent process-tree discovery
  (`cx*/fl*/fl_mpi*` + install-tree matching).
- **Reliable launch** — `launch_fluent` spawns fluent.exe directly with a
  persistent server-owned `-sifile` (no launcher timeout; credentials survive).
- **Background jobs** — `import_geometry` / `generate_surface_mesh` /
  `generate_volume_mesh` return immediately; poll `get_job_status` +
  `read_console_tail` (live transcript parse: iteration, residuals,
  convergence, errors).
- **Sizing units fixed** — sizing values are interpreted in the CAD import
  unit (what the Fluent panel shows), with explicit `units` conversion and
  `panel_shows` read-back.
- **Dual auto/manual modes** for Meshing AND CFD Analysis
  (`set_meshing_mode` / `set_analysis_mode` / `set_workflow_mode`), with
  `propose_mesh_sizing` / `propose_solver_setup` → `apply_solver_plan`;
  switching modes never discards setup.
- **Interactive prompts** — decision-point tools return AskUserQuestion-
  compatible specs (BC types per zone, sizing tiers from the parsed bounding
  box, server-enforced volume-mesh gate).
- **Run lifecycle** — `setup_standard_monitors` (+ history files),
  `read_monitor_history` with plateau verdicts, `initialize_solution`
  checkpointing, `set_autosave`, `assess_convergence` (C1/C2/C3 from
  residuals + KPI plateaus + mass imbalance).
- **Interactive console** — `project_summary` (status + session menu),
  `get_boundary_state` + transactional BC read-back (`verified` flag),
  `create_midplane` (auto extents, plane-normal view),
  `generate_default_results` (one-call post pack: contours + wall pressure
  + KPI/flow-split JSON).
- **PyFluent 0.40.1 compatibility** — StringVector workflow crash
  monkeypatched at the source (re-attach no longer wipes task state),
  result tools rebuilt on `report_definitions.compute()`, protobuf
  health-stub collision fixed.

## Install

1. **Clone or unzip** this folder anywhere, e.g. `C:\path\to\pyfluent-mcp-core\`
   (or `~/pyfluent-mcp-core/` on macOS/Linux).
2. Install dependencies: `pip install -r requirements.txt`

Then point your client at **`pyfluent_mcp_server.py`** directly — it's a
standalone script with its own `mcp = FastMCP(...)` app and `if __name__ ==
"__main__": main()` entry point. No launcher/wrapper needed.

> **Transport note:** the server runs stdio by default (Claude Desktop /
> Cursor / Windsurf). For Streamable HTTP, change the last line of `main()`
> in `pyfluent_mcp_server.py` to `mcp.run(transport="streamable-http")`.

## Claude Desktop config

`%APPDATA%\Claude\claude_desktop_config.json` (Windows) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "pyfluent-mcp-core": {
      "command": "C:\\path\\to\\your\\conda\\envs\\pyansys-env\\python.exe",
      "args": ["C:\\path\\to\\pyfluent-mcp-core\\pyfluent_mcp_server.py"]
    }
  }
}
```

### Guided Fluent launch: cores, GUI, and stray-process cleanup

`launch_fluent` (v3.1.0+) now:
- takes `processor_count` and `show_gui` — its docstring instructs the
  calling assistant to **ask the user** for both before launching, rather
  than silently defaulting;
- takes `kill_existing` (default `True`) — terminates any stray
  `fluent`/`cortex`/`cxsolver` OS processes (via `psutil` if installed, else
  the platform's native `tasklist`/`taskkill` or `ps`/`kill`) before
  launching, so a leftover session can't hold a license slot or get
  connected to by accident.

Two standalone tools give explicit control over the same cleanup:

| Tool | Purpose |
|------|---------|
| `list_fluent_processes()` | list OS-level Fluent processes, including ones this MCP session didn't launch |
| `kill_fluent_processes(pids, force)` | terminate all (or specific) Fluent processes and drop this session's own handles |

## File manifest

```
pyfluent-mcp-core/
├── pyfluent_mcp_server.py        # the whole server — run this directly
├── requirements.txt
├── LICENSE                       # PolyForm Noncommercial 1.0.0
└── README.md
```

## Customize / maintain

- Environment stays on the pinned set: grpcio==1.71.2 (+sub-packages),
  protobuf==5.29.6, ansys-fluent-core==0.40.1 — see the "Customize /
  maintain" note in `requirements.txt`.

## License

Licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

**Free to use** for any noncommercial purpose — personal projects, research,
education, evaluation, internal non-revenue-generating use, etc.

**Commercial use requires a separate license.** If you want to use this
software (or a modified/derivative version of it) for or in connection with
a commercial product, service, or any revenue-generating activity, contact
the copyright holder first to arrange a commercial license:

- Vijay Sankaran
- vijaysankaran0606@gmail.com

## Developer

**Vijay Sankaran**

## Note on provenance

This package is derived from `pyfluent-mcp-v5` (tools layer v3.11.0) by
removing the CFD Intelligence / knowledge-base layer (`kb_router.py`,
`kb_server.py`, `fluent_kb_index.json`, `command_registry.json`,
`postprocessing_menu.json`, `simulation_sops.json`, `split_kb.py`,
`test_kb_package.py`, `knowledge/`, and the `pyfluent_mcp_server_v5.py`
launcher that wired them in). See the full `pyfluent-mcp-v5` package for
that layer's classification/phase-plan/gate tools and ported knowledge base.
