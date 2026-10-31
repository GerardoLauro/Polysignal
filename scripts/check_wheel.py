"""Install a built wheel in a temporary directory and check its CLI away from source."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    distribution = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "dist"
    wheels = list(distribution.glob("polysignal-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Build exactly one wheel before this check")
    with tempfile.TemporaryDirectory(prefix="polysignal-wheel-") as tmp:
        target = Path(tmp) / "installed"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(target), str(wheels[0])], check=True)
        code = """
import pathlib,sys
target=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(target))
import polysignal
assert pathlib.Path(polysignal.__file__).resolve().is_relative_to(target)
from typer.testing import CliRunner
from polysignal.cli.main import app
result=CliRunner().invoke(app,['--help'])
assert result.exit_code == 0, result.output
from polysignal.config import Settings
assert Settings(_env_file=None).data_dir == pathlib.Path('data')
print('Installed wheel and CLI OK outside source tree')
"""
        subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(target)], cwd=tmp, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
