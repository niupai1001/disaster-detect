"""Continue at the first unstarted registered arm, then apply the endpoint screen."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from run_v3_fr import read
from screen import ARMS, screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("release","prior","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    root = args.output/"s6_component_trials"
    before = root/"N0"
    # B0/BP are unaffected by the N0 initialization fix and keep their own original identities.
    if (before/"INITIALIZATION_REVISION_PENDING").exists():
        failure = args.output/"compare_failure.json"
        deadline = time.monotonic()+2100
        while not failure.is_file() and time.monotonic()<deadline:
            time.sleep(3)
        assert read(root/"BP/status.json")["status"]=="COMPLETE"
        assert "FileExistsError" in read(failure)["error"] and not (before/"identity.json").exists()
        archive = args.output/"s6_excluded_zero_update/N0_before_initialization_fix"
        before.rename(archive)
        failure.rename(archive/"compare_failure.json")
    unstarted = [arm for arm in ARMS if not (root/arm/"status.json").exists()]
    if unstarted:
        subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("run_v3_fr.py")),"compare",
            "--release",str(args.release),"--prior",str(args.prior),"--output",str(args.output),
            "--arms",*unstarted],check=True)
    screen(args.output)
    subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("spatial_screen.py")),
                    str(args.output)],check=True)
    if read(root/"screen.json")["status"]=="NO_CANDIDATE":
        from seed_confirmation import run
        run(args)


if __name__=="__main__":
    main()
