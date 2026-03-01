import json
import os
import readline
import shlex
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Optional
from praxis.generator import SkillGenerator, GeneratorConfig
from praxis.executor import SkillExecutor, ExecutionConfig

def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class _C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return "".join(codes) + text + _C.RESET

def _ok(msg: str)   -> None: print(_c(f"✓ {msg}", _C.GREEN))
def _err(msg: str)  -> None: print(_c(f"✗ {msg}", _C.RED), file=sys.stderr)
def _info(msg: str) -> None: print(_c(f"  {msg}", _C.DIM))
def _head(msg: str) -> None: print(_c(msg, _C.BOLD, _C.CYAN))
def _warn(msg: str) -> None: print(_c(f"! {msg}", _C.YELLOW))


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    try:
        import sqlite_vec
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except Exception as e:
        _warn(f"sqlite-vec not available — semantic search disabled: {e}")

    schema_path = Path(__file__).parent / "db" / "schema.sql"
    if schema_path.exists():
        db.executescript(schema_path.read_text())
    else:
        _warn(f"schema.sql not found at {schema_path} — DB may be incomplete")

    return db


def _init_storage(db: sqlite3.Connection, data_dir: Path):
    from praxis.storage import SkillStorage
    return SkillStorage(db=db, data_dir=data_dir)


def _init_llm(provider: str):
    if provider == "anthropic":
        from praxis.llm.anthropic_client import AnthropicClient
        return AnthropicClient()
    elif provider == "openai":
        from praxis.llm.openai_client import OpenAIClient
        return OpenAIClient()
    elif provider == "ollama":
        from praxis.llm.ollama_client import OllamaClient
        url = os.environ.get("PRAXIS_OLLAMA_URL", "http://localhost:11434")
        return OllamaClient(base_url=url)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use anthropic, openai, or ollama.")


def _parse_params(tokens: list[str]) -> dict:
    params = {}
    for token in tokens:
        if "=" not in token:
            _warn(f"Ignoring malformed param (expected key=value): {token!r}")
            continue
        key, _, raw = token.partition("=")
        key = key.strip()
        # Attempt JSON decode so ints/floats/lists pass through correctly
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw  # treat as plain string
    return params


def _cmd_generate(args: list[str], ctx: dict) -> None:
    if not args:
        _err("Usage: generate <natural language request>")
        return

    request = " ".join(args)
    generator = ctx["generator"]

    _info(f"Generating skill for: {request!r}")
    print()

    # Optionally gather hints interactively
    hints = None
    try:
        raw = input(_c("  Input hints (key:type,key:type or leave blank): ", _C.DIM)).strip()
        if raw:
            from praxis.generator import GenerationHints
            input_keys = {}
            for pair in raw.split(","):
                pair = pair.strip()
                if ":" in pair:
                    k, _, t = pair.partition(":")
                    input_keys[k.strip()] = t.strip()
                elif pair:
                    input_keys[pair] = "any"
            hints = GenerationHints(input_keys=input_keys)
    except (EOFError, KeyboardInterrupt):
        print()

    print()
    _info("Calling LLM for schema inference...")

    from praxis.generator import GenerationHints
    result = generator.generate(request, hints=hints or GenerationHints())

    print()
    if not result.success:
        _err(f"Generation failed: {result.error}")
        return

    skill = result.skill

    if result.was_reused:
        _ok(f"Reused existing skill  [{result.reuse_similarity:.0%} match]")
    else:
        _ok(f"Generated in {result.generation_attempts} attempt(s)")

    print()
    _head(f"  {skill.name}  (v{skill.version})")
    _info(f"  {skill.short_desc}")
    if result.version_id:
        _info(f"  version_id : {result.version_id}")
    _info(f"  category   : {skill.category or 'unset'}")

    print()
    print(_c("  Inputs:", _C.BOLD))
    for k, t in skill.inputs_schema.items():
        print(f"    {_c(k, _C.CYAN)} : {t}")

    print(_c("  Outputs:", _C.BOLD))
    for k, t in skill.outputs_schema.items():
        print(f"    {_c(k, _C.CYAN)} : {t}")

    if skill.termination_condition:
        print()
        _info(f"  Done when: {skill.termination_condition}")

    print()
    _info(f"Run it with:  run {skill.name} " +
          " ".join(f"{k}=<{t}>" for k, t in skill.inputs_schema.items()))


def _cmd_run(args: list[str], ctx: dict) -> None:
    if not args:
        _err("Usage: run <skill_name[@version]> [key=value ...]")
        return

    target = args[0]
    param_tokens = args[1:]

    # Parse name@version syntax
    name, _, version_str = target.partition("@")
    version: Optional[int] = None
    if version_str:
        try:
            version = int(version_str)
        except ValueError:
            _err(f"Invalid version {version_str!r} — must be an integer")
            return

    storage  = ctx["storage"]
    executor = ctx["executor"]

    # Load skill
    try:
        skill = storage.load_skill_by_name(name, version=version)
    except ValueError as e:
        _err(str(e))
        _info(f"Tip: generate this skill first with:  generate <description>")
        return

    params = _parse_params(param_tokens)

    # Warn about missing required inputs
    missing = [k for k in skill.inputs_schema if k not in params]
    if missing:
        _warn(f"Missing inputs: {', '.join(missing)}")
        _info("Continuing anyway — skill may fail or use defaults")

    print()
    _info(f"Running {skill.name} v{skill.version} ...")
    _info(f"Params: {json.dumps(params)}")
    print()

    from praxis.skill import ExecutionContext
    context = ExecutionContext(
        user_request=f"CLI run of {name}",
        params=params,
    )

    result = executor.execute(skill, context)

    if result.success:
        _ok(f"Completed in {result.latency_ms}ms")
        print()
        print(_c("  Result:", _C.BOLD))
        _pretty_print_result(result.data or {})
    else:
        _err(f"Execution failed  ({result.latency_ms}ms)")
        print()
        print(_c("  Error:", _C.BOLD))
        print(f"    {result.error}")


def _pretty_print_result(data: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{pad}{_c(k, _C.CYAN)}:")
            _pretty_print_result(v, indent + 2)
        elif isinstance(v, list) and len(v) > 6:
            print(f"{pad}{_c(k, _C.CYAN)}: [{v[0]}, {v[1]}, ... ({len(v)} items)]")
        else:
            print(f"{pad}{_c(k, _C.CYAN)}: {v}")


def _cmd_help() -> None:
    print()
    print(_c("Praxis REPL commands", _C.BOLD))
    print()
    cmds = [
        ("generate <request>",             "Generate a skill from natural language"),
        ("run <name>[@v] [key=val ...]",   "Execute a skill by name, with params"),
        ("help",                           "Show this message"),
        ("exit / quit / Ctrl-D",           "Exit"),
    ]
    for cmd, desc in cmds:
        print(f"  {_c(cmd, _C.CYAN):<40}  {desc}")

    print()
    print(_c("Param types", _C.BOLD))
    print("  Integers and floats are auto-detected: x=42  ratio=0.5")
    print("  JSON values work too:                  items=[1,2,3]")
    print("  Strings with spaces need quoting:      label=\"hello world\"")
    print()
    print(_c("Examples", _C.BOLD))
    print("  generate a skill that sorts a list of numbers")
    print("  run sort_numbers items=[3,1,2]")
    print("  run resize_image@2 width=640 height=480 path=/tmp/img.jpg")
    print()


HISTORY_FILE = Path.home() / ".praxis_history"
PROMPT = _c("praxis> ", _C.BOLD + _C.BLUE) if _supports_color() else "praxis> "

BANNER = """
PRAXIS
"""

class PraxisREPL:
    def __init__(self):
        _load_dotenv()

        provider = os.environ.get("PRAXIS_LLM_PROVIDER", "ollama").lower()
        db_path  = Path(os.environ.get("PRAXIS_DB_PATH",  "~/.praxis/praxis.db")).expanduser()
        data_dir = Path(os.environ.get("PRAXIS_DATA_DIR", "~/.praxis/data")).expanduser()
        threshold = float(os.environ.get("PRAXIS_REUSE_THRESHOLD", "0.75"))

        _info(f"Provider : {provider}")
        _info(f"DB       : {db_path}")
        _info(f"Data dir : {data_dir}")
        print()

        # Init DB
        try:
            db = _init_db(db_path)
        except Exception as e:
            _err(f"DB init failed: {e}")
            sys.exit(1)

        storage = _init_storage(db, data_dir)

        # Init LLM
        try:
            llm = _init_llm(provider)
        except Exception as e:
            _err(f"LLM init failed: {e}")
            _info("Check your API keys and provider config in .env")
            sys.exit(1)


        generator = SkillGenerator(
            llm_client=llm,
            storage=storage,
            config=GeneratorConfig(reuse_similarity_threshold=threshold),
        )
        executor = SkillExecutor(
            llm_client=llm,
            storage=storage,
            config=ExecutionConfig(max_repair_attempts=3),
        )

        self.ctx = {
            "llm":       llm,
            "storage":   storage,
            "generator": generator,
            "executor":  executor,
        }

        # Readline history
        if HISTORY_FILE.exists():
            try:
                readline.read_history_file(str(HISTORY_FILE))
            except Exception:
                pass
        readline.set_history_length(500)

    def run(self) -> None:
        print(_c(BANNER, _C.CYAN))
        _ok("Ready. Type 'help' for commands.")
        print()

        while True:
            try:
                raw = input(PROMPT).strip()
            except EOFError:
                print()
                _info("Bye.")
                break
            except KeyboardInterrupt:
                print()
                continue

            if not raw:
                continue

            try:
                tokens = shlex.split(raw)
            except ValueError as e:
                _err(f"Parse error: {e}")
                continue

            cmd, *args = tokens

            if cmd in ("exit", "quit"):
                _info("Bye.")
                break
            elif cmd == "help":
                _cmd_help()
            elif cmd == "generate":
                try:
                    _cmd_generate(args, self.ctx)
                except KeyboardInterrupt:
                    print()
                    _warn("Generation cancelled.")
            elif cmd == "run":
                try:
                    _cmd_run(args, self.ctx)
                except KeyboardInterrupt:
                    print()
                    _warn("Execution cancelled.")
            else:
                _err(f"Unknown command: {cmd!r}")
                _info("Type 'help' to see available commands.")

            print()

        try:
            readline.write_history_file(str(HISTORY_FILE))
        except Exception:
            pass


def main() -> None:
    try:
        repl = PraxisREPL()
        repl.run()
    except Exception as e:
        _err(f"Fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()