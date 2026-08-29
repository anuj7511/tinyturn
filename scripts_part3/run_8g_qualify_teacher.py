"""
Phase-2 8g -- qualify A0 as teacher against the criteria frozen in the brief (Section 8g). Reads
the outputs of run_8e_padding_counterfactual.py and run_8f_vad_boundary_diagnostic.py and states a
single verdict: A0 does not generate teacher logits for a real distillation run until it passes
BOTH the padding criterion and the VAD-boundary criterion.

Frozen criteria (do not adjust after seeing the numbers -- Section 0's whole point in freezing these
here was to prevent exactly that):

Padding (strict):
  - mean absolute probability change <= 0.02
  - no more than 1% of examples change by more than 0.10
  - decision-flip rate at the calibrated threshold <= 1%

VAD-boundary (looser):
  - no more than 10% of examples change by more than 0.20
  - decision-flip rate at the calibrated threshold <= 5%
  - no more than 2pp degradation in real-audio FCR at fixed complete-turn recall

Usage:
  python scripts_part3/run_8g_qualify_teacher.py
"""
import json
import sys
from pathlib import Path

A0_DIR = Path("experiments") / "A0_whisper_tiny_pv2speechend"
PADDING_PATH = A0_DIR / "8e_padding_counterfactual.json"
VAD_PATH = Path("experiments") / "8f_vad_boundary_diagnostic.json"


def main():
    missing = [p for p in (PADDING_PATH, VAD_PATH) if not p.exists()]
    if missing:
        print("ERROR: missing required inputs, run first:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    padding = json.load(open(PADDING_PATH))["summary"]
    vad = json.load(open(VAD_PATH))["summary"]["A0"]

    padding_pass = (padding["criterion_mean_abs_change_le_0.02"]
                     and padding["criterion_frac_gt_010_le_0.01"]
                     and padding["criterion_flip_rate_le_0.01"])

    vad_checks = []
    vad_notes = []
    for alt_name, entry in vad.items():
        ok = entry["criterion_frac_gt_020_le_0.10"] and entry["criterion_flip_rate_le_0.05"]
        if "criterion_fcr_degradation_le_2pp" in entry:
            ok = ok and entry["criterion_fcr_degradation_le_2pp"]
        else:
            vad_notes.append(f"{alt_name}: {entry.get('real_fcr_note', 'FCR criterion not evaluated')}")
        vad_checks.append((alt_name, ok, entry["n"]))
    vad_pass = all(ok for _, ok, _ in vad_checks) if vad_checks else False

    verdict = {
        "padding_criterion": {"pass": padding_pass, "detail": padding},
        "vad_boundary_criterion": {
            "pass": vad_pass,
            "per_alternative": {name: ok for name, ok, _ in vad_checks},
            "detail": vad,
            "notes": vad_notes,
        },
        "teacher_qualified": bool(padding_pass and vad_pass),
    }

    if vad_notes:
        verdict["caveat"] = (
            "The VAD-boundary check ran on the 206-clip pilot overlap (Phase-2 8f), which is "
            "explicitly a development/pilot set, not a hard-gate-sized sample. If frac_gt_0.20 or "
            "flip_rate look borderline, or the real-audio FCR degradation couldn't be computed at "
            "this n, recompute alternative boundaries on the full 3,000-clip E5 sample (Phase-2 8f) "
            "before treating this qualification as final."
        )

    out_path = A0_DIR / "8g_qualification.json"
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(json.dumps(verdict, indent=2))
    print(f"\nsaved {out_path}")
    print(f"\n{'PASS' if verdict['teacher_qualified'] else 'FAIL'}: A0 is "
          f"{'' if verdict['teacher_qualified'] else 'NOT '}qualified to generate teacher logits "
          f"for a real distillation run.")


if __name__ == "__main__":
    main()
