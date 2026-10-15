"""Helpers shared by the explorer tests."""

from fdsreader.explorer.cli import main


class FakeSim:
    """Just enough of a simulation for the interactive front end when fields and series are patched in."""

    chid = "fake"
    root_path = "."


def run(capsys, *argv):
    """Run the CLI and return (exit status, stdout, stderr)."""
    status = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return status, captured.out, captured.err
