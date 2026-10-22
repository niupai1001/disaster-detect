"""Resolve public git refs without GitHub API, then snapshot exact commit files."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from archive_method_sources_v4r2 import ROOT, REPOS, read

OUT = ROOT / "reports/v4r2_methods/sources_pinned"


def capture(item):
    repo, (branch, files) = item
    result = {"repo": repo, "branch": branch, "commit": None,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    try:
        completed = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", f"https://github.com/{repo}.git", f"refs/heads/{branch}"],
                                   capture_output=True, text=True, timeout=45, check=True)
        commit = completed.stdout.split()[0]
        if len(commit) != 40:
            raise ValueError("Invalid git ref")
        result["commit"] = commit
    except Exception as exc:
        result["commit_error"] = str(exc)
        return result
    for name in files:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{name}"
        record = {"path": name, "url": url}
        try:
            data = read(url)
            path = OUT / repo.replace("/", "__") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise FileExistsError("Refuse pinned snapshot overwrite")
            path.write_bytes(data)
            record.update(local_path=path.relative_to(ROOT).as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as exc:
            record["error"] = str(exc)
        result["files"].append(record)
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(capture, REPOS.items()))
    path = ROOT / "reports/v4r2_methods/source_index_pinned.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=1)
    print(json.dumps(results, indent=1))
