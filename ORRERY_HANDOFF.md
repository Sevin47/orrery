# ORRERY — Project Handoff

> A gamified project/task tracker rendered as a personal galaxy. Every project is a planet; every completed task is a physical fragment of that planet. This document is the complete handoff for continuing development in Claude Code.

---

## 1. Product Vision

The owner wants a **beautiful, serene, aesthetically-driven browser game** that tracks their real work. It is not a parody of work and not a dashboard — it is a slow, cosmic, contemplative space that makes finished work feel permanent and visible.

Core fantasy: **your career becomes a galaxy you can fly through.**

### Decisions locked in with the owner (do not change without asking)

| Question | Decision |
|---|---|
| Format | Web browser game |
| Data | Real projects and tasks from the owner's actual job |
| Structure | Projects are the "something"; tasks are the pieces that build it |
| Intake | At intake, the player chooses: new project OR task for an existing project |
| Planet interaction | Selecting a planet **splits it apart** into fragments (exploded view); each fragment = one completed task; clicking a fragment shows that task's info |
| Incomplete tasks | Orbiting **stardust** that **crystallizes** into the planet when completed |
| Task fields | Title, notes, due date, status |
| Planet appearance | Mix of auto-generated (deterministic from project identity) + evolves with task count/size |
| Art direction | Serene & cosmic — deep blues, nebulas, ambient. NOT neon/synthwave, NOT cartoonish |
| Navigation | Click/zoom camera glides. No ship, no free-fly |
| Persistence | **Forever, unless the user deletes it themselves** |
| Project completion | Quiet celebration: bloom of light + gentle chime (not a fireworks show) |
| Audio | Generative ambient audio, toggleable |

---

## 2. Current State

A working v1 prototype exists as a single-file React component: **`orrery.jsx`** (included alongside this document). It was built for the Claude.ai artifact environment and runs there today. All mechanics below are implemented and functional.

### Implemented features

- **Galaxy view**: planets on a golden-angle spiral layout, slowly spinning, starfield (~2,200 stars) + additive nebula sprites, exponential fog, mouse-parallax camera drift, hover tooltip with project name.
- **Planet view**: camera glides to the planet; the planet **explodes** into per-task fragments (eased lerp, spin slows while exploded); hover glow on fragments; clicking a fragment opens the task detail panel; leaving the view reassembles the planet.
- **Stardust**: each open task is a 64-particle additive point cluster orbiting at its own radius/speed/inclination with twinkle. Clicking one opens the detail panel with a **Crystallize** action.
- **Crystallize animation**: dust spirals inward over ~1.35s, then the planet rebuilds with the new fragment, a light bloom expands, and a chime plays. If it was the final task: bigger white bloom + soft 3-note chord (C5/G5/E6) and a "world complete" tag.
- **Planet generation** (deterministic per project id): hue in the teal→blue→violet range (HSL hue base 195–325/360), latitude banding, polar caps, scattered bright "cities of light" faces, radius grows with completed count, atmosphere glow sprite, **rings appear at ≥8 crystallized tasks**, newborn projects render as a small pulsing molten core.
- **Intake modal**: two tabs — "New world" (name + description) and "New task" (world selector, title, notes, due date). After creating a project it auto-switches to task intake with that project preselected.
- **Task actions**: crystallize (complete), return to stardust (un-complete), delete task, delete world (with inline confirm).
- **Index panel**: collapsible list of all worlds with done/total counts; click to fly there.
- **Persistence**: full state saved (debounced 600ms) to `window.storage` under key `orrery_galaxy_v1` as JSON `{ projects: [...] }`. "saved" note flashes in the corner.
- **Ambient audio** (Tone.js): sine PolySynth through a 14s-decay reverb playing random pentatonic-ish notes (C3–G4 set) every ~5.2s; separate triangle synth for chimes. Off by default; toggle button starts the AudioContext (user gesture).
- **Accessibility/quality**: `prefers-reduced-motion` slows rotation, disables parallax and the dust spiral animation; focus-visible outlines; labeled form fields; responsive panels (detail panel docks to bottom on ≤640px).

### Known limitations / not yet built

- No task **editing** after creation (only complete/uncomplete/delete).
- No overdue visual treatment in the 3D scene (overdue only shows as amber text in the detail panel).
- Planet positions shift when a project is deleted (layout is index-based, not id-stable).
- Hover raycasts run on every pointermove (fine at current scale; throttle if planet count grows large).
- No data export/import, no undo, no multi-device sync beyond the storage backend.
- Single-file component (~1,100 lines) — should be modularized during the port.

---

## 3. Architecture (as built)

Single default-exported React component `Orrery` with Three.js managed imperatively via refs (React never re-renders the canvas).

### Data model

```js
Project {
  id: string (uuid),
  name: string,
  desc: string,
  createdAt: number (ms epoch),
  tasks: Task[]
}
Task {
  id: string (uuid),
  title: string,
  notes: string,
  due: "YYYY-MM-DD" | null,
  done: boolean,
  createdAt: number,
  completedAt: number | null
}
```

Storage payload: `JSON.stringify({ projects })` at key `orrery_galaxy_v1`.

### Three.js scene graph (per planet "record")

```
Group (position = layoutPos(index))
├─ spin: Group (rotates; y-angle stored on record)
│   └─ chunk meshes  — one Mesh per completed task
│       geometry: faces of IcosahedronGeometry(R, 3) bucketed by
│                 nearest of N fibonacci-sphere seed directions
│       material: MeshStandardMaterial, vertexColors, flatShading,
│                 per-chunk instance (hover sets .emissive)
│       exploded offset = seedDir * explodeT * (0.55R + 0.45)
│   └─ OR molten core mesh when 0 tasks complete (emissive, pulsing)
├─ dust clusters (children of Group, NOT spin) — one per open task
│   ├─ THREE.Points (64 pts, additive, orbit computed per-frame)
│   └─ invisible click sphere (r=0.5) carrying userData.dustTaskId
├─ atmosphere glow Sprite (additive radial-gradient canvas texture)
├─ ring Mesh (RingGeometry) when completedCount >= 8
└─ invisible hit sphere (r = R + 0.8) carrying userData.projectId
```

Record fields: `{ group, spin, chunks[], dust[], atmo, ring, hit, R, pal, sig, explode, spinAngle, spinSpeed }`.

### Key mechanisms

- **Determinism**: `strHash` (FNV-1a) + `mulberry32` PRNG seeded from project/task ids drive palette, banding, orbit params, spin speed. A planet always looks the same for the same data.
- **Scene sync**: `useEffect([projects])` diffs a `Map<projectId, record>` against state. A per-project `signature(p)` (`id | taskId+doneFlag list`) decides rebuild vs. reposition. Rebuilds preserve `explode` and `spinAngle` so the exploded view doesn't snap.
- **Crystallize sequencing**: click → dust record gets `anim = {t: 0}` → render loop spirals it inward → `setTimeout(1350ms)` → React state marks task done → sync rebuilds planet → bloom sprite + chime spawn ~60ms later. Under reduced motion, state updates immediately.
- **Camera**: `camTarget`/`lookTarget` vectors lerped every frame (`dt * 2.2` / `dt * 2.6`). Galaxy target scales out with project count (`z = min(110, 30 + n*6.5)`). Planet target = planet pos + `(0, 0.9R, 3.4R + 4.2)`.
- **Picking**: one `Raycaster`; galaxy mode tests planet hit-spheres, planet mode tests focused planet's chunks + dust click-spheres.
- **Blooms**: transient sprites in a `world.blooms` array, scale/opacity animated in the loop, disposed on completion.
- **Disposal**: `disposeRecord` traverses and disposes geometries/materials/textures on every rebuild/removal; full teardown on unmount.

### Design system

```
Background   #04060f          Fog: FogExp2(0x04060f, 0.0075)
Ink          #e9edff          Dim ink: #9aa6cf
Panels       rgba(9,14,34,.74) + blur(10px), 1px border rgba(150,175,255,.16)
Accent       #9db8ff          Warn (overdue): #e8c58a
Planet hues  HSL hue 0.54–0.90 (teal → blue → violet), never warm
Display font 'Cormorant Garamond' (brand, planet/task titles, tooltips — italic)
Body font    'Outfit' 300–500
Signature    the exploded-planet interaction; keep everything else quiet
Buttons      pill-shaped; primary = soft blue gradient with faint outer glow
Motion       slow (spin ~0.05–0.10 rad/s), eased lerps, nothing snappy
```

Copy voice: cosmic but plain — "Birth this world", "Send into orbit", "Crystallize", "Return to stardust", "world complete". Keep this vocabulary consistent everywhere.

---

## 4. Porting Notes (important)

1. **`window.storage` is a Claude.ai artifact API.** It does not exist outside that environment. When porting, replace the two call sites (one `get` on mount, one debounced `set`) with `localStorage`/IndexedDB for a purely local app, or a small backend/DB if the owner wants sync. Preserve the "forever unless deleted" contract — never expire or auto-prune data, and keep explicit delete confirmation for worlds.
2. **Three.js version**: written against r128 idioms (no OrbitControls used — camera is fully custom; keep it that way, the glide is a design decision). If upgrading three, re-verify `IcosahedronGeometry` non-indexed behavior and `Points`/`Sprite` material options.
3. **Fonts** load via Google Fonts `@import` inside a `<style>` tag; move to proper `<link>`/self-hosting in a real build.
4. **Tone.js** requires a user gesture before `Tone.start()`; the sound toggle is that gesture. Keep audio off by default.
5. Suggested modularization: `lib/rng.js`, `lib/planetFactory.js`, `lib/textures.js`, `scene/World.js` (loop, camera, picking), `state/store.js` (data + persistence), `ui/` (panels, intake modal), `App.jsx`.

---

## 5. Backlog (owner-approved directions, roughly prioritized)

1. **Task editing** — edit title/notes/due from the detail panel.
2. **Overdue in-scene treatment** — overdue stardust tinted subtly warmer (amber) with a slightly faster shimmer; stays serene.
3. **Stable planet layout** — derive position from project id hash instead of array index so worlds never move.
4. **Import from real tools** — Todoist/Asana/Jira/Linear import or webhook so intake can be automatic. (Owner said tasks come from their real job; this is the highest-leverage future feature.)
5. **Constellations** — optional lines/tags linking related projects.
6. **Completed-world gallery mode** — a slow auto-orbit "museum" pass over finished planets.
7. **Data export/import** — JSON download/restore as a safety net.
8. **Performance** — throttle hover raycasts; instanced dust if task counts get large; pause render loop when tab hidden.
9. **Nice-to-haves discussed but not committed**: XP/levels, streaks (owner never confirmed wanting streak pressure — ask before adding), day/era log per planet.

---

## 6. Acceptance Criteria for Any Change

- The serene cosmic aesthetic is preserved: cool palette only, slow motion, additive glows, no UI clutter over the scene.
- The exploded-planet interaction remains the signature; don't replace it with menus.
- Determinism holds: same data ⇒ same galaxy.
- Nothing is ever deleted without an explicit user action + confirmation for worlds.
- Reduced-motion users get a functional, calmer equivalent of every animation.
- All geometry/material/texture created at runtime is disposed when replaced.

---

*Handoff prepared July 13, 2026. Companion file: `orrery.jsx` (v1 prototype, single-file React + Three.js + Tone.js).*
