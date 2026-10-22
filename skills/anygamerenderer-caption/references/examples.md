# Example specs

Each block is a complete `spec.json`. All were rendered with the shipped model; the first three ship as `examples/`.

## First-person shooter (weapon proxy + enemy group)

The large green cylinder pinned to the lower right is the player's rifle (`many: false`, viewpoint-locked motion); every
blue cylinder is one rival cadet (`many: true`).

```json
{
  "shape": "cylinder",
  "setting": "a futuristic indoor training arena: smooth white and pale-gray curved walls, dark teal panels, red floor stripes and hazard markings, raised platforms and elevated walkways with railings under a vaulted dark ceiling, lit by cool white light with teal and pink neon accents",
  "subjects": [
    {"color": "green", "desc": "the player's own weapon seen in first person: a futuristic blue-and-magenta assault rifle with white accents and a glowing emitter, held in brown-gloved hands at the lower right of the frame", "many": false, "motion": "locked to the viewpoint, swaying gently with each step"},
    {"color": "blue", "desc": "a rival cadet in a dark-gray combat suit with red trim and a visored helmet, carrying a rifle", "many": true, "motion": "moving as recorded"}
  ],
  "sound": "hum of arena ventilation, footsteps on hard flooring, electronic rifle fire echoing"
}
```

## Top-down crowd (allied squad + enemy ranks)

Two groups, both `many: true`. The cyan squad is a tight cluster (the inspect script reports it as one blob — still a
group); describing both uniforms with distinct colors keeps the sides apart.

```json
{
  "shape": "wire",
  "setting": "a container port seen from high above: a long pale-gray concrete dock road with white lane arrows and pedestrian stripes, lined on both sides by stacked shipping containers in teal, yellow and gray, under a hazy white sky",
  "subjects": [
    {"color": "cyan", "desc": "a blue-uniformed soldier with a dark helmet and rifle, jogging in tight formation with identical squadmates", "many": true, "motion": "moving as recorded"},
    {"color": "blue", "desc": "a red-uniformed enemy infantryman with a dark helmet and rifle, standing or charging in ranks", "many": true, "motion": "moving as recorded"}
  ],
  "sound": "hundreds of boots on concrete, shouted orders, distant harbor cranes"
}
```

## Vehicle race (one color for all karts)

All karts share one proxy color, so one `many: true` subject covers the player and the rivals; the description says
which is which by camera behavior.

```json
{
  "shape": "box",
  "setting": "a sunlit asphalt race circuit: smooth dark tarmac with white edge lines and red-white curbs, tire barriers, grandstands and advertising boards, green verges and distant hills under a clear sky",
  "subjects": [
    {"color": "orange", "desc": "a sleek open-wheel racing kart with a low body, exposed engine and a helmeted driver; the one the camera follows is the player's kart, the others are rival karts in different liveries", "many": true, "motion": "driving exactly along its recorded path"}
  ],
  "sound": "screaming kart engines, tires squealing on asphalt, crowd noise"
}
```

## Ball sport (athletes, positive-only wording)

Proxies only slide, so the animation request goes into `motion`; the broadcast look is stated positively instead of
"no HUD".

```json
{
  "shape": "cylinder",
  "setting": "a football stadium: a green gridiron with white yard lines, hash marks and yard numbers, a blue HOME end zone and an orange AWAY end zone, yellow goal posts at both ends, grandstands full of spectators along both sidelines, tall floodlight masts and a pale blue sky, shown as a clean live-action-quality sports broadcast",
  "subjects": [
    {"color": "green", "desc": "an American football player of the blue home team in a bright blue jersey with white numbers, white pants and a blue helmet, a fully articulated athlete", "many": true, "motion": "running with full motion-captured strides, arms pumping and torso leaning into turns, never sliding or frozen"},
    {"color": "blue", "desc": "an American football player of the orange away team in an orange jersey, white pants and an orange helmet, a fully articulated athlete", "many": true, "motion": "running with full motion-captured strides, arms pumping and torso leaning into turns, never sliding or frozen"}
  ],
  "sound": "crowd roar, whistles, pads colliding, a stadium announcer"
}
```

## Stylized (LEGO version of the shooter)

The art direction goes in `style`; each subject's `desc` ends with its re-imagining; the setting gets the materials
sentence. Sentences copied from `styles.md`.

```json
{
  "shape": "cylinder",
  "setting": "a futuristic indoor training arena: smooth white and pale-gray curved walls, dark teal panels, red floor stripes, raised platforms and elevated walkways under a vaulted dark ceiling, with every structure, floor and railing assembled from LEGO bricks, plates and tiles with visible studs",
  "subjects": [
    {"color": "green", "desc": "the player's own weapon seen in first person, rebuilt as a LEGO brick blaster: a chunky gray plastic brick gun held by yellow claw hands at the lower right", "many": false, "motion": "locked to the viewpoint, swaying gently with each step"},
    {"color": "blue", "desc": "a rival cadet in a dark-gray combat suit with red trim, rebuilt as a LEGO minifigure: cylindrical yellow head with a printed face, claw-shaped hands, blocky torso with the suit printed on it, stiff plastic legs, glossy ABS plastic", "many": true, "motion": "moving as recorded"}
  ],
  "sound": "plastic clicks, footsteps on hard flooring, toy-like rifle sounds",
  "style": "a LEGO movie: the entire world is built from real LEGO bricks with visible studs, seams and the glossy ABS plastic sheen, lit like a studio-shot toy diorama (The LEGO Movie look), slight stop-motion feel, bright primary colors"
}
```
