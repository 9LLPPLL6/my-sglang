# AGENTS.md — SGLang

## Repo structure
- `python/sglang/srt/` — core runtime (scheduler `managers/scheduler.py:1`, tokenizer `managers/tokenizer_manager.py:1`, `model_executor/model_runner.py:1`, `server_args.py:1`, `environ.py:1`, `runtime_context.py:1`)
- `python/sglang/kernels/` — `aot/` (heavyweight AOT, `sgl-kernel`) and `jit/` (lightweight JIT, see `add-jit-kernel` skill)
- `rust/` — Cargo workspace (`sglang-server`, `sglang-grpc`, `sglang-mm`); Python bindings auto-discovered via `setup.py` + `[package.metadata.sglang]` in each crate
- `sgl-model-gateway/` — Rust gateway (own Cargo workspace, `make` targets)
- `experimental/sgl-router/` — separate Rust service
- `python/pyproject.toml:1` — main package (`setuptools` + `setuptools-rust` + `setuptools-scm` via `scripts/release/get_version_tag.py:1` → `python/sglang/_version.py:1`); other `pyproject.toml` under `rust/*/`, `sgl-model-gateway/`, `3rdparty/`
- `test/registered/` — CI tests (auto-discovered by `test/run_suite.py:1`), `test/manual/` — local-only, `benchmark/` — perf scripts, `docs/` — Mintlify site (`docs/AGENTS.md:1` for docs conventions)

## Setup & build
- Editable install (CUDA): `pip install -e python` (builds Rust extensions; needs `protoc`, Rust toolchain, CUDA). Platform variants restrict extensions via `[tool.sglang] rust-extensions` in `pyproject_other.toml`.
- CI install path: `scripts/ci/cuda/ci_install_dependency.sh:1` (reference for deps, `sgl-kernel` wheel handling)
- Env vars: defined in `python/sglang/srt/environ.py:1` as `EnvField` on `envs`; name `SGLANG_*`, access via `envs.FOO.get()` (not `os.environ`). See `env-var-conventions` skill.
- Version fallback: `0.0.0.dev0` when `.git` unavailable (`python/pyproject.toml:248`).

## Lint / format (must pass `lint.yml:1`)
- Run: `pre-commit run --all-files --show-diff-on-failure` (or `SKIP=no-commit-to-branch pre-commit run --all-files` as CI does)
- Stack: `ruff --select F401,F821,UP037 --fix` + `isort` (profile `black`, `known_first_party=sglang` in `.isort.cfg:1`) + `black-jupyter` + `clang-format` (C++/CUDA) + `codespell` + `rustfmt`/`clippy` for `rust/` + `cargo fmt` for gateway/router
- Excludes: `python/sglang/srt/grpc/*_pb2*.py`, `python/sglang/kernels/aot/*`, `rust` handled separately — don't fix generated gRPC stubs
- Also gated: `docs/scripts/check_cookbook_configs.mjs:1` and `mint broken-links --check-anchors --check-redirects` in `docs/` (Mintlify `mint@4.2.559`)

## Testing — single file / single test
```bash
# single file / single method (unittest or pytest; runner appends -f failfast — don't add argparse/sys.argv hacks)
python3 test/registered/core/test_srt_endpoint.py
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# via CI runner (discovers registered/ + jit kernel files)
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small --auto-partition-id 0 --auto-partition-size 4

# JIT kernel single test
python3 test/registered/jit/test_add_constant.py
```

## Testing — writing / registration
- Base class: `sglang.test.test_utils.CustomTestCase` (never `unittest.TestCase`) — ensures `tearDownClass` runs even if `setUpClass` fails. `tearDownClass` must guard with `hasattr(cls,"process")` before `kill_process_tree`.
- Server fixtures: `popen_launch_server` + `kill_process_tree` from `sglang.srt.utils` / `sglang.test.test_utils`, or inherit `DefaultServerBase` (`python/sglang/test/server_fixtures/default_fixture.py:1`). Reuse one server per file via `setUpClass`.
- Prefer mock (`unittest.mock.patch`/`MagicMock`) in `test/registered/unit/` when no real inference needed; mirror `python/sglang/srt/` layout.
- Registration (AST-parsed — `est_time`/`stage`/`runner_config`/`suite` must be **literals**):
  ```python
  from sglang.test.ci.ci_register import register_cuda_ci
  register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")  # → suite base-b-test-1-gpu-small
  # or legacy: register_cuda_ci(est_time=80, suite="base-b-test-1-gpu-small")
  ```
  Only add `register_amd_ci`/`register_npu_ci` for backend-specific paths; common tests use CUDA only.
- End every file with `if __name__ == "__main__": unittest.main()` (or `pytest.main([__file__])`) — no extra args (runner appends `-f`).
- Suites: `scripts/ci/runner_configs.yml:1` maps `runner_config` → runner label; stage naming is `{stage}-test-{runner_config}`. CUDA nightly uses `stage="nightly"` (no `nightly=True`); other backends use `nightly=True` flag.
- Full checklist/suite tables: `test/README.md:1` and `.claude/skills/write-sglang-test/SKILL.md:1`.

## CI pipeline (`pr-test.yml:1`, `.claude/skills/ci-workflow-guide/SKILL.md:1`)
- Flow: `Lint` → `check-changes` (path filter) → `pr-gate` → stages `base-a` (~3min) → `base-b` (~30min) → `base-c` (~30min); `sgl-kernel`/`jit-kernel`/`multimodal-gen` run parallel to `base-b`. Scheduled (`00 11,23 UTC`) runs all stages in parallel.
- Sequential gating on PRs via `wait-for-base-a`/`wait-for-base-b` (poll GitHub API); bypass with `bypass-fastfail` label. `pr-test-finish` aggregates — omitted job silently greens run.
- `check-changes` sets `main_package`/`sgl_kernel`/`jit_kernel`/`multimodal_gen` and `partitions` (auto-partition by `est_time`, cap ~30min/job). `rust-ext-build` always runs (cache key = source hash).
- Debugging: rerun via `gh workflow run pr-test.yml` or label `run-ci`/`rerun-failed`; inspect `test/run_suite.py` partitioning.

## Conventions & gotchas
- Code style: `.claude/rules/general-code-style.md:1` — stateless/pure fns, immutable defaults, extract init-static values in `__init__` (e.g. `self.mtp_enabled`), fns <100 LOC, files <2k LOC, prefer `protected` + keyword args, don't pass god objects (`Scheduler`/`ModelRunner`) — pass specific values. Also: `no-dataclasses.md`, `no-getattr-defensive.md`, `schedule-batch-out-of-place-mutation.md`, `large-class-style.md` (for `Scheduler`/`TokenizerManager`/`ModelRunner`).
- `docs_new/` is forbidden (renamed to `docs/` in #32123) — pre-commit `forbid-docs-new-path` and lint gate (`lint.yml:23`) check `git ls-files -- docs_new`.
- Don't create `test/registered/xpu/test_nvidia_nemotron_3_nano.py` — gitignored (too large for CI).
- `python/sglang/kernels/aot/` excluded from sdist/wheel (`python/pyproject.toml:214`) and lint `files:` filter.
- Issue/PR: `CODE_OF_CONDUCT.md:1`, `.github/labeler.yml:1`, `.github/CI_PERMISSIONS.json:1` for rerun perms.

## Key skill references
- `write-sglang-test` — test templates, model constants (`DEFAULT_SMALL_MODEL_NAME_FOR_TEST` etc. in `python/sglang/test/test_utils.py:104`), fixtures, suite selection
- `ci-workflow-guide` — stage ordering, gating, partitioning, execution modes
- `add-jit-kernel` / `add-sgl-kernel` — kernel contribution flows
- `sglang-runtime-context` — `ServerArgs`/`RuntimeContext` tiers before touching global state
