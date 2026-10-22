"""Build an AnyGameRenderer caption from a short JSON spec.

The model was trained on captions with a fixed six-section layout (MiniMax-H3 "video editing" format).
Write the spec, not the caption: this script fills the layout exactly as in training.

Usage:
    python make_caption.py spec.json                 # caption to stdout
    python make_caption.py spec.json -o caption.txt
    python make_caption.py --template                # print an example spec

spec.json:
{
  "shape": "cylinder",          # proxy shape used in the white model: box | wire | cylinder | sphere
  "setting": "a futuristic indoor training arena: smooth white curved walls, ...",   # one sentence, no trailing period
  "subjects": [                 # one entry per proxy color, in the order you want them numbered
    {"color": "green", "desc": "the player's own weapon seen in first person: a futuristic assault rifle ...",
     "many": false, "motion": "locked to the viewpoint, swaying gently with each step"},
    {"color": "blue",  "desc": "a rival cadet in a dark-gray combat suit with a visored helmet, carrying a rifle",
     "many": true,  "motion": "moving as recorded"}
  ],
  "sound": "hum of arena ventilation, footsteps on hard flooring, rifle fire echoing",
  "style": null,                # optional; default = photoreal AAA. Keep it to lighting / weather / grade, never geometry
  "occlusion": true             # optional; adds the clause that blockout volumes stay opaque (recommended)
}
  color   color name of the proxy as it appears in the white model: red, orange, yellow, yellow-green, green, cyan, blue, purple, magenta
  many    true  = every proxy of that color is one instance of this subject (crowds, enemies, rival karts)
          false = exactly one proxy of that color is this subject (the player, the first-person weapon)
  motion  how the subject moves on top of the proxy's path; typical values:
          "moving as recorded"  |  "driving exactly along its recorded path"  |  "locked to the viewpoint, swaying gently with each step"
"""
import argparse
import json
import sys

SHAPE_WORD = {"box": "box", "wire": "wireframe box", "cylinder": "cylinder", "sphere": "ellipsoid"}
DEFAULT_STYLE = ("a polished, photoreal AAA game visual style: physically based materials, crisp micro-detail, soft global illumination, "
                 "filmic colour grading, subtle depth of field and motion blur")
VID1 = ("<Video 1> is the source video for the target video edit. It is an untextured blockout capture of this exact take: pale grey primitive geometry for the whole environment "
        "and plain coloured {S}s, one per character or vehicle, sliding along their exact paths, rendered with the exact camera motion of the take.")
RET_V = ("<Video 1> (source video editing): partially_preserved - its camera trajectory, scene layout, object placement and temporal structure are preserved exactly; every grey blockout surface is "
         "re-rendered in place as the corresponding real-world surface with the same silhouette, scale and position, and every coloured {S} reads as its finished subject in every frame, "
         "at the same position, size and heading, appearing exactly when its {S} appears and vanishing exactly when its {S} vanishes; areas empty in <Video 1> stay empty, and no character, "
         "vehicle or creature is added that has no {S} in <Video 1>.")
OCCL = (" Every grey blockout volume of <Video 1>, including any that passes close to or in front of the camera, is rendered as solid opaque geometry of the setting at exactly the same place and size, "
        "occluding whatever lies behind it exactly as it does in <Video 1>; nothing is seen through blockout geometry, and no surface that is hidden in <Video 1> becomes visible.")
CONSISTENCY = " Consistency with <Video 1> takes priority over everything else: geometry, camera and characters follow <Video 1> frame by frame."

TEMPLATE = {
    "shape": "cylinder",
    "setting": "a futuristic indoor training arena: smooth white and pale-grey curved walls, dark teal panels, red floor stripes, raised platforms and elevated walkways with railings under a vaulted dark ceiling, lit by cool white light with teal and pink neon accents",
    "subjects": [
        {"color": "green", "desc": "the player's own weapon seen in first person: a futuristic blue-and-magenta assault rifle with white accents and a glowing emitter, held in brown-gloved hands at the lower right of the frame", "many": False, "motion": "locked to the viewpoint, swaying gently with each step"},
        {"color": "blue", "desc": "a rival cadet in a dark-grey combat suit with red trim and a visored helmet, carrying a rifle", "many": True, "motion": "moving as recorded"},
    ],
    "sound": "hum of arena ventilation, footsteps on hard flooring, electronic rifle fire echoing",
    "style": None,
    "occlusion": True,
}


def make_caption(spec):
    S = SHAPE_WORD[spec["shape"]]
    style = spec.get("style") or DEFAULT_STYLE
    occl = OCCL if spec.get("occlusion", True) else ""
    subs, rets, shots = [], [], []
    for n, s in enumerate(spec["subjects"], 1):
        who, col, verb = f"<Subject {n}>", s["color"], s.get("motion", "moving as recorded")
        if s.get("many"):
            subs.append(f"{who} is {s['desc']}; every {col} {S} in <Video 1> is one such subject, and the whole {col} group is a group of identical ones.")
            rets.append(f"{who}: partially_preserved - the position, apparent size, count, heading and recorded movement of every {col} {S} from <Video 1> are preserved exactly, while each flat {col} {S} is replaced by one subject {verb}.")
            shots.append(f"Each {col} {S} becomes one {who}, exactly along its recorded path and timing, no more and no fewer than the {S}s present in each frame.")
        else:
            subs.append(f"{who} is {s['desc']}; represented in <Video 1> by the {col} {S}.")
            rets.append(f"{who}: partially_preserved - its position, apparent size, heading and recorded movement from <Video 1> are preserved exactly, while the flat {col} {S} is replaced by the subject {verb}.")
            shots.append(f"The {col} {S} becomes {who}, exactly along its recorded path and timing.")
    return ("subject_definitions:\n" + "\n".join(subs) + "\n" + VID1.format(S=S)
            + f"\n\nsummary:\n[video editing] The target video is an edited version of <Video 1>. The flat blockout visuals are replaced with a polished, photoreal AAA game render. The target setting is {spec['setting']}."
            + "\n\nretention_analysis:\n" + "\n".join(rets) + "\n" + RET_V.format(S=S) + occl
            + f"\n\ndetailed_description:\nThe target video is in {style}.{CONSISTENCY}{occl}\n"
            + "[Shot 1] The shot follows <Video 1> as one continuous view; the camera repeats the recorded motion exactly. Every grey blockout block of <Video 1> becomes the matching structure of the setting in the same place, with the same silhouette and scale, and the grey ground becomes the setting's ground. "
            + " ".join(shots) + " Nothing moves that does not move in <Video 1>."
            + f"\n\noverall_soundscape:\n{spec['sound']}\n\nnon_diegetic_music:\nN/A")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--template", action="store_true", help="print an example spec and exit")
    a = ap.parse_args()
    if a.template:
        print(json.dumps(TEMPLATE, indent=2))
        return
    if not a.spec:
        ap.error("spec.json required (or --template)")
    cap = make_caption(json.load(open(a.spec)))
    if a.out:
        open(a.out, "w").write(cap + "\n")
    else:
        sys.stdout.write(cap + "\n")


if __name__ == "__main__":
    main()
