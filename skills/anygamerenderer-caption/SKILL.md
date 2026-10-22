---
name: anygamerenderer-caption
description: Write the spec.json / caption that AnyGameRenderer needs to render a crude gameplay recording (gray blockout with colored proxy shapes) into a finished game video, then run the render. Use this whenever the user has a white-model / blockout / primitive video and wants it rendered, asks how to describe their game or characters for AnyGameRenderer, wants a specific art style (LEGO, Ghibli, pixel art, synthwave, …), or mentions make_caption.py, spec.json, jobs.jsonl, or infer.py — even if they just say "render this clip".
---

# Writing AnyGameRenderer captions

AnyGameRenderer renders a crude recording — pale-gray geometry with flat-colored proxy shapes sliding along the true
paths of characters, weapons, and vehicles — into a game video. The video gives it *where* and *when*; the caption gives it
*what* and *how it looks*. A good caption therefore does two things: binds every proxy color to a concrete subject, and
describes the setting and art style precisely. You never write the caption by hand: you write a small `spec.json` and
`make_caption.py` expands it into the six-section layout the model was trained on (any other layout degrades sync).

## Workflow

1. **Look at the input.** Run `python skills/anygamerenderer-caption/scripts/inspect_primitive.py clip/primitive.mp4`:
   it reports the proxy colors present (by name), how many proxies of each color appear, the clip length and the
   frame count the model will generate. Also open a few frames (the script writes a contact sheet) to see the proxy
   shape — box / wireframe box / cylinder / ellipsoid — and the camera (first person with a weapon proxy locked at the
   lower right, third person, top down, chase cam).
2. **Ask or infer what each color is.** The user usually knows ("blue = enemy soldiers, green = my rifle"). If not,
   infer from behavior: a large proxy fixed at the lower right of a first-person view is the player's weapon; one proxy
   the camera follows is the player; a group of identical proxies is enemies, allies, cars, or athletes.
3. **Write `spec.json`** (fields below), run `python make_caption.py clip/spec.json -o clip/caption.txt`, and read the
   caption once to check nothing contradicts the video.
4. **Render**: add a line to `jobs.jsonl` and run `CUDA_VISIBLE_DEVICES=0 python infer.py jobs.jsonl` (≈ 17 min per 5 s
   clip; finished outputs are skipped on re-run). Check the output against the input for sync before iterating on style.

## spec.json fields

```json
{
  "shape": "cylinder",
  "setting": "a futuristic indoor training arena: smooth white curved walls, dark teal panels, red floor stripes, raised platforms under a vaulted dark ceiling, cool white light with teal and pink neon accents",
  "subjects": [
    {"color": "green", "desc": "the player's own weapon seen in first person: a futuristic blue-and-magenta assault rifle held in gloved hands at the lower right of the frame", "many": false, "motion": "locked to the viewpoint, swaying gently with each step"},
    {"color": "blue",  "desc": "a rival cadet in a dark-gray combat suit with red trim and a visored helmet, carrying a rifle", "many": true, "motion": "moving as recorded"}
  ],
  "sound": "hum of arena ventilation, footsteps on hard flooring, electronic rifle fire echoing",
  "style": null,
  "occlusion": true
}
```

| field | what to write |
|---|---|
| `shape` | the proxy shape used in this clip: `box`, `wire` (wireframe box), `cylinder` or `sphere` (ellipsoid). One shape per clip. |
| `setting` | one concrete sentence of what the gray geometry becomes: ground, walls / buildings, props, sky, and lighting. Name materials and colors; it is the only description of the environment the model gets. |
| `subjects[].color` | the proxy color *as it appears in the video*, from: red, orange, yellow, yellow-green, green, cyan, blue, purple, magenta. Use the script's names; a color that is not in the video binds nothing. |
| `subjects[].desc` | who/what the proxy is, concretely: role, outfit, or body, colors, what it carries. Start with the role ("the player's own weapon seen in first person: …", "a rival kart …"). |
| `subjects[].many` | `true` when every proxy of that color is one instance of the same subject (crowds, enemies, rival cars, athletes); `false` when exactly one proxy is this subject (the player, the first-person weapon). Wrong `many` is the most common cause of missing or duplicated entities. |
| `subjects[].motion` | how the subject moves on top of the proxy's path: `moving as recorded` (default), `driving exactly along its recorded path` (vehicles), `locked to the viewpoint, swaying gently with each step` (first-person weapon). Add animation hints here ("running with full strides, arms pumping") rather than in `desc`. |
| `sound` | a short soundscape line; keep it consistent with the action (no gunfire sounds in a racing game). |
| `style` | optional art direction; omit for photoreal AAA. See *Styles*. |
| `occlusion` | leave `true`; it keeps blockout volumes opaque so nothing is seen through walls. |

## Rules that matter for sync

- **One subject per color, every color bound.** Each distinct proxy color in the video needs exactly one subject
  entry; unbound proxies are rendered as whatever the model guesses, or stay as blocks.
- **Describe only what is in the video.** Do not add characters, vehicles, or effects that have no proxy; the model will
  draw them and they will not track anything. Effects a game really has (muzzle flash, boost flames, ball in flight)
  may be mentioned in `desc`/`motion`, but only if that event actually happens in the clip.
- **Positive statements only.** The model has no negative guidance: "no HUD", "no gunfire", "no text" tend to *produce*
  a HUD, gunfire, or text. Say what is there, not what is not. For ball sports write "a clean live-action-quality sports
  broadcast" instead of "no HUD markers".
- **Setting must match the geometry.** A top-down blockout of a road grid should be described as a road grid, not a
  forest; the model keeps the layout and will fight a mismatched setting with drifting cameras.
- **Keep the proxy wording to the script.** Do not mention "box", "cylinder", "blockout" or colors inside `desc`;
  `make_caption.py` writes the proxy binding sentences itself, in the form the model was trained on.

## Styles

The default is a polished photoreal AAA look. For another look, put the **art direction** in `style` and the
**character re-imagining** at the end of each subject's `desc`, because the model only restyles characters when the
subject description says so:

```json
"style": "a LEGO movie: the entire world is built from real LEGO bricks with visible studs and the glossy ABS plastic sheen, lit like a studio-shot toy diorama, bright primary colors",
"subjects": [{"color": "blue", "desc": "a rival cadet in a dark-gray combat suit with red trim, rebuilt as a LEGO minifigure: cylindrical yellow head with a printed face, claw hands, blocky torso with the suit printed on it", "many": true, "motion": "moving as recorded"}]
```

Three sentences make a style work: *medium/look* (what the whole frame is: a LEGO diorama, a hand-painted Ghibli
background, a 16-bit pixel-art game, a sumi-e ink wash), *materials of the setting* (bricks / gouache / pixel tiles /
ink washes for ground, walls, sky) and *character re-imagining* per subject (minifigure / cel-shaded figure / sprite /
brush strokes, keeping the identity colors). Styles that re-skin the world (samurai Japan, Star Wars, cyberpunk,
Fallout, candy land, steampunk) and painterly media (oil, watercolor, crayon, stained glass) take well; flat 2D media
(pixel art, paper cutout, pop art, chalk) are harder and may come out as a tinted 3D render. Keep lighting / weather /
time of day inside `style` too; never put geometry there. `references/styles.md` has ready-made style + character
sentences for 30 looks — copy them rather than inventing wording.

## Examples

`references/examples.md` holds complete specs for the common cases: first-person shooter with a weapon proxy, top-down
crowd with allies and enemies, vehicle race, ball sport, and a stylized (LEGO) version of the first one. Start from the
closest one.
