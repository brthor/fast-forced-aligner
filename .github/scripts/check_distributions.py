"""Check the installed-package boundary and required distribution resources."""

from email.parser import BytesParser
from pathlib import Path
import tarfile
import tomllib
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    wheels = list((root / "dist").glob("*.whl"))
    sources = list((root / "dist").glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Expected one wheel and one source archive"
    package_files = {
        "fast_forced_aligner/__init__.py",
        "fast_forced_aligner/alignment.py",
        "fast_forced_aligner/csrc/alignment.cu",
        "fast_forced_aligner/py.typed",
    }
    licenses = project["license-files"]
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        actual = {name for name in names if ".dist-info/" not in name}
        assert actual == package_files, f"Unexpected wheel package contents: {actual}"
        metadata_path, = (name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        assert metadata["Version"] == project["version"]
        assert metadata["License-Expression"] == project["license"]
        assert set(metadata.get_all("License-File", [])) == set(licenses)
        prefix = metadata_path.removesuffix("METADATA") + "licenses/"
        for name in licenses:
            assert wheel.read(prefix + name) == (root / name).read_bytes(), name
        for name in package_files:
            assert wheel.read(name) == (root / "src" / name).read_bytes(), name

    with tarfile.open(sources[0]) as source:
        names = source.getnames()
        prefix, = {name.split("/")[0] for name in names}
        required = {
            "README.md", "CITATION.cff", "pyproject.toml",
            "examples/transformers_ctc/align.py",
            "examples/transformers_ctc/requirements.txt",
            "examples/transformers_ctc/Dockerfile",
            "benchmarks/patches/cpu_corrected.cpp",
            *licenses,
            *("src/" + name for name in package_files),
        }
        for name in required:
            member = source.extractfile(prefix + "/" + name)
            assert member is not None, name
            assert member.read() == (root / name).read_bytes(), name
        source_package = {
            member.name.removeprefix(prefix + "/src/")
            for member in source.getmembers()
            if member.isfile() and member.name.startswith(prefix + "/src/fast_forced_aligner/")
        }
        assert source_package == package_files, source_package
        assert not any("/.local/" in name or "/__pycache__/" in name for name in names)
    print("Wheel and source archive contain the core package, CUDA source, and licensing resources.")


if __name__ == "__main__":
    main()
