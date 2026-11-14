"""Save the exact first comparison sources from their bound hashes."""
from hashlib import sha256
import json
from pathlib import Path
import sys


campaign = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
bindings = [("B0","source_revision_b0_bp"),("N0","source_revision_n0_through_c")]
confirmation_replacements = {
    "scripts/audit_trials.py": [
        (b"    if not path.exists():\n        return\n", b""),
    ],
    "scripts/compare_trials.py": [
        (b"            elif args.seed!=17 and arm==selected[\"candidate\"]:\n"
         b"                parent = root/f\"{selected['baseline']}_seed{args.seed}\"\n"
         b"                assert read(parent/\"status.json\")[\"status\"]==\"COMPLETE\"\n"
         b"                old = read(parent/\"identity.json\")[\"initial_parameters\"]\n",b""),
        (b"            combination = arm==\"N0\" or (args.seed!=17 and options(arm).frontend!=options(selected[\"baseline\"]).frontend)\n"
         b"            common = {key:names[key] for key in old.keys() & names.keys()\n"
         b"                      if not (combination and key.startswith(\"encoder.spectral.\"))}\n",
         b"            common = {key:names[key] for key in old.keys() & names.keys()\n"
         b"                      if not (arm==\"N0\" and key.startswith(\"encoder.spectral.\"))}\n")
    ],
    "scripts/spatial_screen.py": [
        (b"                    for observation in scene_geometry:\n"
         b"                        marker = f\"{observation}_{task}_\"\n"
         b"                        group = [k.removesuffix(\"target_mask\") for k in base.files\n"
         b"                                 if k.startswith(marker) and k.endswith(\"_target_mask\")]\n"
         b"                        if group:\n"
         b"                            groups[observation] = sorted(group)\n",
         b"                    for field in sorted(k for k in base.files if k.endswith(\"_target_mask\") and f\"_{task}_\" in k):\n"
         b"                        prefix = field.removesuffix(\"target_mask\")\n"
         b"                        observation = prefix.split(f\"_{task}_\")[0]\n"
         b"                        groups.setdefault(observation,[]).append(prefix)\n"),
    ],
}
initial_replacements = {
    "scripts/compare_trials.py": [
        (b"        registered = list(parents)[:-1]\n"
         b"        assert args.arms==registered[registered.index(args.arms[0]):]\n"
         b"        for earlier in registered[:registered.index(args.arms[0])]:\n"
         b"            assert read(root/earlier/\"status.json\")[\"status\"]==\"COMPLETE\"\n",
         b"        assert args.arms==[\"B0\",\"BP\",\"N0\",\"N1\",\"N2\",\"C\"]\n"),
    ],
    "src/encoder_v3/film_resnet/fusion.py": [
        (b"        if blocks not in (0,2) or (film and blocks!=2):\n"
         b"            raise ValueError(\"Registered frontend requires zero or two blocks\")\n"
         b"        # Initialize the common heads identically to the full candidate before removing the ablated blocks.\n"
         b"        self.frontend = Frontend(2,film)\n",b"        self.frontend = Frontend(blocks,film)\n"),
        (b"        if blocks==0:\n            self.frontend.blocks = nn.ModuleList()\n",b""),
    ],
}
for arm,name in bindings:
    binding = json.loads((campaign/f"s6_component_trials/{arm}/identity.json").read_bytes())
    target = campaign/name
    target.mkdir(exist_ok=True)
    for rel,expected in binding["candidate_files"].items():
        data = (root/rel).read_bytes()
        for variants in ([confirmation_replacements,initial_replacements] if arm=="B0" else [confirmation_replacements]):
            if sha256(data).hexdigest()==expected:
                break
            for before,after in variants.get(rel,[]):
                assert data.count(before)==1,rel
                data = data.replace(before,after)
        assert sha256(data).hexdigest()==expected,rel
        path = target/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    print("Exact source revision",arm,len(binding["candidate_files"]),"files")
