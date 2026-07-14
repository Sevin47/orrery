import React, { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import * as Tone from "tone";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";

/* ---------------- deterministic helpers ---------------- */
function strHash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function fibSphere(n) {
  const pts = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = n === 1 ? 0 : 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = ga * i;
    pts.push(new THREE.Vector3(Math.cos(th) * r, y, Math.sin(th) * r).normalize());
  }
  return pts;
}
// Lays every project out along a golden-angle spiral, but — unlike a fixed
// per-index step — grows the orbital radius by each planet's own footprint
// as it goes. That keeps huge (Jupiter-scale) worlds from ever overlapping
// their neighbors' hit-spheres (which was causing clicks near a big planet's
// dust to land on a different, nearby planet), and the slow, low-frequency
// vertical drift reads as a calm flowing curve instead of a jagged zigzag.
// A planet's visual footprint reaches well past its body: rings sit at 2.15R,
// and the additive atmosphere glow's meaningfully-bright zone runs to ~2.3R
// (it fades further past that). Spacing needs to clear THIS, not the small
// click hit-sphere, or halos/rings wash into each other at large sizes.
const PLANET_VISUAL_SPREAD = 2.6;

function computeLayout(projects) {
  const ga = 2.39996;
  const positions = [];
  let orbitR = 0;
  let prevR = 0;
  for (let i = 0; i < projects.length; i++) {
    const R = radiusFor(projects[i]);
    if (i === 0) {
      positions.push(new THREE.Vector3(0, 0, 0));
      prevR = R;
      continue;
    }
    const gap = 8; // flat breathing room on top of the size-scaled clearance below
    orbitR = Math.max(orbitR + (prevR + R) * PLANET_VISUAL_SPREAD + gap, 18);
    const a = i * ga;
    const y = Math.sin(i * 0.27 + 0.6) * (2.6 + R * 0.35);
    positions.push(new THREE.Vector3(Math.cos(a) * orbitR, y, Math.sin(a) * orbitR));
    prevR = R;
  }
  return positions;
}
function paletteFor(project) {
  const h = strHash(project.id);
  const hue = (195 + (h % 130)) / 360; // teal -> blue -> violet
  return {
    hue,
    sat: 0.42 + ((h >> 8) % 100) / 400,
    accent: new THREE.Color().setHSL((hue + 0.08) % 1, 0.7, 0.72),
    glow: new THREE.Color().setHSL(hue, 0.8, 0.66),
  };
}
function radiusFor(project) {
  const done = project.tasks.filter((t) => t.done).length;
  const total = project.tasks.length;
  // power-curve growth spanning roughly an Earth-to-Jupiter ratio (~11x) between
  // a freshly-started world and a long-running one, before tapering off.
  const growth = Math.pow(done, 0.72) * 1.05 + Math.log1p(total) * 0.15;
  return 1.0 + Math.min(15, growth);
}
function signature(p) {
  return p.id + "|" + p.tasks.map((t) => t.id + (t.done ? "1" : "0")).join(",");
}
function uid() {
  return (crypto?.randomUUID?.() || Date.now() + "-" + Math.random().toString(36).slice(2));
}
function fmtDate(d) {
  if (!d) return null;
  const dt = new Date(d + "T00:00:00");
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}
function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}
function randSphereDir(rng) {
  const z = rng() * 2 - 1;
  const th = rng() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return new THREE.Vector3(r * Math.cos(th), z, r * Math.sin(th));
}
function makeNoiseAxes(rng, n, freqBase) {
  const axes = [];
  for (let i = 0; i < n; i++) {
    axes.push({
      ax: randSphereDir(rng),
      freq: freqBase * (1 + i * 0.65 + rng() * 0.35),
      phase: rng() * Math.PI * 2,
      amp: 1 / (i + 1),
    });
  }
  return axes;
}
// smooth, continuous, deterministic pseudo-noise on the unit sphere (sum of
// phase-shifted sine waves over random great-circle axes — no seams, no grain)
function sphereFBM(dir, axes) {
  let sum = 0, norm = 0;
  for (let i = 0; i < axes.length; i++) {
    const a = axes[i];
    sum += a.amp * Math.sin(dir.dot(a.ax) * a.freq * Math.PI + a.phase);
    norm += a.amp;
  }
  return norm ? sum / norm : 0;
}
function vertexNormalsFromPositions(posAttr) {
  const vCount = posAttr.count;
  const normals = new Float32Array(vCount * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < vCount; i++) {
    v.fromBufferAttribute(posAttr, i).normalize();
    normals[i * 3] = v.x; normals[i * 3 + 1] = v.y; normals[i * 3 + 2] = v.z;
  }
  return normals;
}

// icosahedron subdivision (faces = 20*(detail+1)^2). Fragment boundaries follow
// individual triangle edges, so this also sets how "toothy" a split seam looks —
// detail 15 was ~4.3° per tooth (visibly jagged when exploded); 60 is ~1.1°.
const PLANET_DETAIL = 60;

// IcosahedronGeometry + mergeVertices is expensive (the dominant cost of a planet
// rebuild) but topology-only — every planet uses the same detail level, and R just
// scales it. Build the indexed unit sphere once and reuse its index/directions for
// every planet, scaling positions by R per-build instead of re-deriving them.
let _unitSphereCache = null;
function getUnitSphere() {
  if (_unitSphereCache) return _unitSphereCache;
  const merged = mergeVertices(new THREE.IcosahedronGeometry(1, PLANET_DETAIL));
  const pos = merged.attributes.position;
  const vCount = pos.count;
  const dirs = new Float32Array(vCount * 3);
  for (let i = 0; i < dirs.length; i++) dirs[i] = pos.array[i]; // already unit length (R=1)
  const index = merged.index.array.slice(); // plain typed array, topology only
  merged.dispose();
  _unitSphereCache = { dirs, index, vCount, faceCount: index.length / 3 };
  return _unitSphereCache;
}

/* ---------------- textures ---------------- */
function glowTexture(rgb = "160,190,255") {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const gr = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  gr.addColorStop(0, `rgba(${rgb},0.85)`);
  gr.addColorStop(0.35, `rgba(${rgb},0.30)`);
  gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  return t;
}

/* ---------------- planet construction ---------------- */
function buildPlanetRecord(project, pos) {
  const pal = paletteFor(project);
  const rng = mulberry32(strHash(project.id));
  const R = radiusFor(project);
  const doneTasks = project.tasks.filter((t) => t.done);
  const openTasks = project.tasks.filter((t) => !t.done);
  const group = new THREE.Group();
  group.position.copy(pos);
  const spin = new THREE.Group();
  group.add(spin);
  const chunks = [];

  const bandPhase = rng() * Math.PI * 2;
  const banding = 2.5 + rng() * 3;
  const capSize = 0.72 + rng() * 0.16;
  const continentAxes = makeNoiseAxes(rng, 5, 1.6);
  const mountainAxes = makeNoiseAxes(rng, 3, 4.2);
  const craterCount = 3 + Math.floor(rng() * 5);
  const craters = [];
  for (let i = 0; i < craterCount; i++) {
    craters.push({ center: randSphereDir(rng), radius: 0.14 + rng() * 0.20 });
  }

  // per-vertex terrain: smooth continents/oceans (sat+light, never hue into warm),
  // sparse craters with a shadowed bowl + bright rim, latitude bands, polar caps,
  // scattered "cities of light". dir must be a unit vector.
  const terrainColor = (dir) => {
    const lat = Math.asin(clamp(dir.y, -1, 1));
    const continent = sphereFBM(dir, continentAxes); // ~[-1,1]
    const mountain = sphereFBM(dir, mountainAxes);
    const landAmt = smoothstep(-0.05, 0.25, continent);
    const band = Math.sin(lat * banding + bandPhase) * 0.5;

    let hue = pal.hue + (continent > 0 ? 0.012 : -0.014) + mountain * 0.006;
    let sat = pal.sat + (continent > 0 ? -0.06 : 0.14) + band * 0.05;
    let light = 0.32 + continent * 0.13 + mountain * landAmt * 0.09 + band * 0.07;

    for (let i = 0; i < craters.length; i++) {
      const c = craters[i];
      const d = Math.acos(clamp(dir.dot(c.center), -1, 1)) / c.radius;
      if (d < 1) {
        light += -0.20 * (1 - smoothstep(0, 0.55, d)) + 0.18 * Math.exp(-Math.pow((d - 0.82) / 0.12, 2));
        sat *= 0.9;
      }
    }

    if (Math.abs(lat) > capSize * (Math.PI / 2)) { light += 0.26; sat *= 0.5; }

    const sparkle = mulberry32(strHash(project.id + dir.x.toFixed(2) + dir.y.toFixed(2) + dir.z.toFixed(2)))();
    if (continent > 0 && sparkle > 0.965) light += 0.16; // scattered "cities of light"

    const col = new THREE.Color();
    col.setHSL(((hue % 1) + 1) % 1, clamp(sat, 0.15, 0.92), clamp(light, 0.08, 0.88));
    return col;
  };

  if (doneTasks.length === 0) {
    // newborn molten core — smooth sphere with mottled lava-noise coloring
    const coreR = 0.85;
    const geo = new THREE.IcosahedronGeometry(coreR, 8);
    const posAttr = geo.attributes.position;
    const vCount = posAttr.count;
    const colors = new Float32Array(vCount * 3);
    const lavaAxes = makeNoiseAxes(rng, 4, 2.4);
    const baseCol = new THREE.Color().setHSL(pal.hue, 0.55, 0.30);
    const hotCol = new THREE.Color().setHSL((pal.hue + 0.03) % 1, 0.85, 0.62);
    const vTmp = new THREE.Vector3();
    for (let i = 0; i < vCount; i++) {
      vTmp.fromBufferAttribute(posAttr, i).normalize();
      const t = smoothstep(-0.2, 0.6, sphereFBM(vTmp, lavaAxes));
      const col = baseCol.clone().lerp(hotCol, t);
      colors[i * 3] = col.r; colors[i * 3 + 1] = col.g; colors[i * 3 + 2] = col.b;
    }
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(vertexNormalsFromPositions(posAttr), 3));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      emissive: pal.glow, emissiveIntensity: 0.5,
      roughness: 0.55, metalness: 0.12,
    });
    const core = new THREE.Mesh(geo, mat);
    core.userData.isCore = true;
    spin.add(core);
    chunks.push({ mesh: core, dir: new THREE.Vector3(0, 1, 0), task: null, isCore: true });
  } else {
    // build off the shared unit-sphere topology (see getUnitSphere), scaled by R,
    // then bucket its FACES by nearest task seed — each fragment keeps a slice of
    // the shared vertex data, so lighting/color stay continuous across fragment
    // seams when reassembled.
    const seeds = fibSphere(doneTasks.length);
    const unit = getUnitSphere();
    const vCount = unit.vCount;
    const baseColors = new Float32Array(vCount * 3);
    const vTmp = new THREE.Vector3();
    for (let i = 0; i < vCount; i++) {
      vTmp.set(unit.dirs[i * 3], unit.dirs[i * 3 + 1], unit.dirs[i * 3 + 2]);
      const col = terrainColor(vTmp);
      baseColors[i * 3] = col.r; baseColors[i * 3 + 1] = col.g; baseColors[i * 3 + 2] = col.b;
    }

    const buckets = seeds.map(() => ({ map: new Map(), positions: [], normalsArr: [], colorsArr: [], indices: [] }));
    const cen = new THREE.Vector3();
    const d0 = new THREE.Vector3(), d1 = new THREE.Vector3(), d2 = new THREE.Vector3();
    for (let f = 0; f < unit.faceCount; f++) {
      const i0 = unit.index[f * 3], i1 = unit.index[f * 3 + 1], i2 = unit.index[f * 3 + 2];
      d0.set(unit.dirs[i0 * 3], unit.dirs[i0 * 3 + 1], unit.dirs[i0 * 3 + 2]);
      d1.set(unit.dirs[i1 * 3], unit.dirs[i1 * 3 + 1], unit.dirs[i1 * 3 + 2]);
      d2.set(unit.dirs[i2 * 3], unit.dirs[i2 * 3 + 1], unit.dirs[i2 * 3 + 2]);
      cen.copy(d0).add(d1).add(d2).normalize();
      let best = 0, bd = -Infinity;
      for (let s = 0; s < seeds.length; s++) {
        const d = cen.dot(seeds[s]);
        if (d > bd) { bd = d; best = s; }
      }
      const b = buckets[best];
      const tri = [i0, i1, i2].map((orig) => {
        let li = b.map.get(orig);
        if (li === undefined) {
          li = b.positions.length / 3;
          b.map.set(orig, li);
          b.positions.push(unit.dirs[orig * 3] * R, unit.dirs[orig * 3 + 1] * R, unit.dirs[orig * 3 + 2] * R);
          b.normalsArr.push(unit.dirs[orig * 3], unit.dirs[orig * 3 + 1], unit.dirs[orig * 3 + 2]);
          b.colorsArr.push(baseColors[orig * 3], baseColors[orig * 3 + 1], baseColors[orig * 3 + 2]);
        }
        return li;
      });
      b.indices.push(tri[0], tri[1], tri[2]);
    }

    seeds.forEach((dir, i) => {
      const b = buckets[i];
      if (b.indices.length === 0) return;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(b.positions, 3));
      geo.setAttribute("normal", new THREE.Float32BufferAttribute(b.normalsArr, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(b.colorsArr, 3));
      geo.setIndex(b.indices);
      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.78, metalness: 0.06, emissive: 0x000000,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.taskId = doneTasks[i] ? doneTasks[i].id : null;
      spin.add(mesh);
      chunks.push({ mesh, dir: dir.clone(), task: doneTasks[i] || null });
    });
  }

  // atmosphere glow
  const atmo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture(`${Math.round(pal.glow.r * 255)},${Math.round(pal.glow.g * 255)},${Math.round(pal.glow.b * 255)}`),
    transparent: true, opacity: doneTasks.length ? 0.5 : 0.32,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  const aScale = (doneTasks.length ? R : 0.9) * 4.6;
  atmo.scale.set(aScale, aScale, 1);
  group.add(atmo);

  // rings for mature worlds
  let ring = null;
  if (doneTasks.length >= 8) {
    const rg = new THREE.RingGeometry(R * 1.55, R * 2.15, 72);
    const rm = new THREE.MeshBasicMaterial({
      color: pal.accent, side: THREE.DoubleSide, transparent: true, opacity: 0.20,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    ring = new THREE.Mesh(rg, rm);
    ring.rotation.x = Math.PI / 2 - 0.35 - rng() * 0.3;
    group.add(ring);
  }

  // stardust clusters for open tasks — soft round cloud-puffs, not hard squares
  const dust = [];
  const dustMap = glowTexture(`${Math.round(pal.accent.r * 255)},${Math.round(pal.accent.g * 255)},${Math.round(pal.accent.b * 255)}`);
  openTasks.forEach((task, i) => {
    const trng = mulberry32(strHash(task.id));
    const count = 72;
    const arr = new Float32Array(count * 3);
    const colArr = new Float32Array(count * 3);
    for (let k = 0; k < count; k++) {
      const rr = Math.pow(trng(), 0.6) * 0.42;
      const th = trng() * Math.PI * 2, ph = Math.acos(2 * trng() - 1);
      arr[k * 3] = rr * Math.sin(ph) * Math.cos(th);
      arr[k * 3 + 1] = rr * Math.cos(ph) * 0.7;
      arr[k * 3 + 2] = rr * Math.sin(ph) * Math.sin(th);
      const jitter = 0.7 + trng() * 0.55; // uneven brightness — reads as wispy dust, not a uniform blob
      colArr[k * 3] = pal.accent.r * jitter;
      colArr[k * 3 + 1] = pal.accent.g * jitter;
      colArr[k * 3 + 2] = pal.accent.b * jitter;
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    pg.setAttribute("color", new THREE.Float32BufferAttribute(colArr, 3));
    const pm = new THREE.PointsMaterial({
      map: dustMap, vertexColors: true, size: 0.24, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });
    const points = new THREE.Points(pg, pm);
    const clickSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.5, 8, 8),
      new THREE.MeshBasicMaterial({ visible: false })
    );
    clickSphere.userData.dustTaskId = task.id;
    const cluster = new THREE.Group();
    cluster.add(points, clickSphere);
    group.add(cluster);
    dust.push({
      cluster, points, clickSphere, task,
      orbitR: R + 1.35 + (i % 5) * 0.55,
      speed: 0.12 + trng() * 0.12,
      phase: trng() * Math.PI * 2,
      incline: (trng() - 0.5) * 0.9,
      anim: null, // {t} while crystallizing
    });
  });

  // invisible galaxy-level click sphere
  const hit = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(R, 1.1) + 0.8, 10, 10),
    new THREE.MeshBasicMaterial({ visible: false })
  );
  hit.userData.projectId = project.id;
  group.add(hit);

  return {
    group, spin, chunks, dust, atmo, ring, hit,
    R, pal, sig: signature(p2s(project)),
    explode: 0, spinAngle: Math.random() * Math.PI * 2,
    spinSpeed: 0.05 + rng() * 0.05,
  };
}
function p2s(p) { return p; }

function disposeRecord(rec) {
  rec.group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    }
  });
}

/* ---------------- main component ---------------- */
export default function Orrery() {
  const mountRef = useRef(null);
  const worldRef = useRef(null);
  const projectsRef = useRef([]);
  const [projects, setProjects] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [view, setView] = useState({ mode: "galaxy", projectId: null });
  const [selected, setSelected] = useState(null); // {projectId, taskId}
  const [hoverTip, setHoverTip] = useState(null);
  const [intakeOpen, setIntakeOpen] = useState(false);
  const [intakeMode, setIntakeMode] = useState("project");
  const [indexOpen, setIndexOpen] = useState(false);
  const [soundMode, setSoundMode] = useState("off"); // "off" | "ambient" | "classical"
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [confirmDeleteTask, setConfirmDeleteTask] = useState(null);
  const [saveNote, setSaveNote] = useState("");
  const viewRef = useRef(view);
  viewRef.current = view;
  const soundRef = useRef({ ready: false, synth: null, chime: null, timer: null });
  const dragRef = useRef({ down: false, moved: false, startX: 0, startY: 0, startTheta: 0, startAlpha: 0 });
  const reducedMotion = useRef(
    typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );

  useEffect(() => {
    setConfirmDeleteTask(null);
  }, [selected]);

  // form state
  const [fProjName, setFProjName] = useState("");
  const [fProjDesc, setFProjDesc] = useState("");
  const [fTaskProject, setFTaskProject] = useState("");
  const [fTaskTitle, setFTaskTitle] = useState("");
  const [fTaskNotes, setFTaskNotes] = useState("");
  const [fTaskDue, setFTaskDue] = useState("");

  /* ---------- persistence (localStorage) ---------- */
  useEffect(() => {
    try {
      const raw = localStorage.getItem("orrery_galaxy_v1");
      if (raw) {
        const data = JSON.parse(raw);
        if (Array.isArray(data.projects)) setProjects(data.projects);
      }
    } catch (e) { /* first visit — nothing saved yet */ }
    setLoaded(true);
  }, []);
  useEffect(() => {
    projectsRef.current = projects;
    if (!loaded) return;
    const t = setTimeout(() => {
      try {
        localStorage.setItem("orrery_galaxy_v1", JSON.stringify({ projects }));
        setSaveNote("saved");
        setTimeout(() => setSaveNote(""), 1500);
      } catch (e) {
        setSaveNote("save failed");
      }
    }, 600);
    return () => clearTimeout(t);
  }, [projects, loaded]);

  /* ---------- audio ---------- */
  const chime = useCallback((big = false) => {
    const s = soundRef.current;
    if (!s.ready) return;
    try {
      const now = Tone.now();
      if (big) {
        s.chime.triggerAttackRelease("C5", "2n", now, 0.5);
        s.chime.triggerAttackRelease("G5", "2n", now + 0.18, 0.4);
        s.chime.triggerAttackRelease("E6", "1n", now + 0.36, 0.3);
      } else {
        const notes = ["A5", "E6", "C6", "D6"];
        s.chime.triggerAttackRelease(notes[Math.floor(Math.random() * notes.length)], "4n", now, 0.35);
      }
    } catch (e) {}
  }, []);

  const stopAmbientLoop = useCallback(() => {
    const s = soundRef.current;
    if (s.ambientTimer) { clearInterval(s.ambientTimer); s.ambientTimer = null; }
  }, []);
  const startAmbientLoop = useCallback(() => {
    const s = soundRef.current;
    if (s.ambientTimer) return; // already running
    const scale = ["C3", "D3", "F3", "G3", "A3", "C4", "D4", "F4", "G4"];
    s.ambientTimer = setInterval(() => {
      try {
        const n = scale[Math.floor(Math.random() * scale.length)];
        s.synth.triggerAttackRelease(n, "2n");
        if (Math.random() < 0.35) {
          const n2 = scale[Math.floor(Math.random() * scale.length)];
          s.synth.triggerAttackRelease(n2, "2n", Tone.now() + 1.2);
        }
      } catch (e) {}
    }, 5200);
  }, []);

  // "Flying"-inspired classical mode: a slow, sparsely bowed solo voice over a
  // soft sustained drone — long tones with minimal movement, like a lone string
  // instrument outdoors, rather than the ambient mode's plucked pentatonic bells.
  const stopClassicalLoop = useCallback(() => {
    const s = soundRef.current;
    if (s.classicalTimer) { clearTimeout(s.classicalTimer); s.classicalTimer = null; }
    if (s.droneOn) {
      try { s.drone.triggerRelease(); } catch (e) {}
      s.droneOn = false;
    }
  }, []);
  const startClassicalLoop = useCallback(() => {
    const s = soundRef.current;
    if (s.classicalTimer) return; // already running
    const scale = ["D3", "F3", "G3", "A3", "C4", "D4", "F4", "A4"]; // D dorian-ish, cello/bass range
    if (!s.droneOn) {
      try { s.drone.triggerAttack("D2"); s.droneOn = true; } catch (e) {}
    }
    const playPhrase = () => {
      try {
        const n = scale[Math.floor(Math.random() * scale.length)];
        const dur = 3 + Math.random() * 3.5;
        s.bow.triggerAttackRelease(n, dur);
      } catch (e) {}
      s.classicalTimer = setTimeout(playPhrase, 9000 + Math.random() * 9000);
    };
    playPhrase();
  }, []);

  const initAudio = useCallback(async () => {
    const s = soundRef.current;
    if (s.ready) return;
    await Tone.start();
    const reverb = new Tone.Reverb({ decay: 14, wet: 0.6 }).toDestination();
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: { attack: 3.5, decay: 2, sustain: 0.35, release: 7 },
    }).connect(reverb);
    synth.volume.value = -22;
    const chimeSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.005, decay: 1.4, sustain: 0.05, release: 2.5 },
    }).connect(reverb);
    chimeSynth.volume.value = -14;

    const classicalReverb = new Tone.Reverb({ decay: 20, wet: 0.55 }).toDestination();
    const bowFilter = new Tone.Filter({ type: "lowpass", frequency: 1000, Q: 0.5 }).connect(classicalReverb);
    const bowVibrato = new Tone.Vibrato({ frequency: 3.6, depth: 0.06 }).connect(bowFilter);
    const bowSynth = new Tone.Synth({
      oscillator: { type: "fatsawtooth", count: 3, spread: 18 },
      envelope: { attack: 3.2, decay: 1.4, sustain: 0.72, release: 6.5 },
    }).connect(bowVibrato);
    bowSynth.volume.value = -18;
    const droneSynth = new Tone.Synth({
      oscillator: { type: "sine" },
      envelope: { attack: 4, decay: 0, sustain: 1, release: 5 },
    }).connect(classicalReverb);
    droneSynth.volume.value = -28;

    s.synth = synth; s.chime = chimeSynth; s.reverb = reverb;
    s.bow = bowSynth; s.drone = droneSynth;
    s.classicalReverb = classicalReverb; s.bowFilter = bowFilter; s.bowVibrato = bowVibrato;
    s.ready = true;
  }, []);

  const cycleSound = useCallback(async () => {
    try { await initAudio(); } catch (e) { return; }
    // side effects live here, not inside setSoundMode's updater — React 18
    // StrictMode double-invokes functional updaters in dev, which was firing
    // startClassicalLoop() twice in the same tick and crashing Tone's scheduler.
    const order = ["off", "ambient", "classical"];
    const next = order[(order.indexOf(soundMode) + 1) % order.length];
    stopAmbientLoop();
    stopClassicalLoop();
    if (next === "ambient") startAmbientLoop();
    else if (next === "classical") startClassicalLoop();
    setSoundMode(next);
  }, [soundMode, initAudio, stopAmbientLoop, stopClassicalLoop, startAmbientLoop, startClassicalLoop]);

  useEffect(() => () => {
    const s = soundRef.current;
    stopAmbientLoop();
    stopClassicalLoop();
    try {
      s.synth?.dispose(); s.chime?.dispose(); s.reverb?.dispose();
      s.bow?.dispose(); s.drone?.dispose();
      s.classicalReverb?.dispose(); s.bowFilter?.dispose(); s.bowVibrato?.dispose();
    } catch (e) {}
  }, [stopAmbientLoop, stopClassicalLoop]);

  /* ---------- three.js world ---------- */
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const scene = new THREE.Scene();
    // Lowered from 0.0075: planets now space out much further apart as they grow
    // (see PLANET_VISUAL_SPREAD), and the old density fogged out most of the
    // galaxy overview once a few large planets pushed the framing distance up.
    scene.fog = new THREE.FogExp2(0x04060f, 0.0028);
    const camera = new THREE.PerspectiveCamera(52, mount.clientWidth / mount.clientHeight, 0.1, 600);
    camera.position.set(0, 34, 96);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x04060f, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x30395c, 1.1));
    const sun = new THREE.DirectionalLight(0xdfe8ff, 1.15);
    sun.position.set(30, 40, 20);
    scene.add(sun);
    const rim = new THREE.PointLight(0x6a7dff, 0.5, 200);
    rim.position.set(-40, -10, -30);
    scene.add(rim);

    // starfield
    const starGeo = new THREE.BufferGeometry();
    const starN = 2200;
    const sp = new Float32Array(starN * 3);
    const sc = new Float32Array(starN * 3);
    for (let i = 0; i < starN; i++) {
      const r = 140 + Math.random() * 180;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      sp[i * 3] = r * Math.sin(ph) * Math.cos(th);
      sp[i * 3 + 1] = r * Math.cos(ph);
      sp[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      const c = new THREE.Color().setHSL(0.55 + Math.random() * 0.15, 0.5, 0.6 + Math.random() * 0.35);
      sc[i * 3] = c.r; sc[i * 3 + 1] = c.g; sc[i * 3 + 2] = c.b;
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
    starGeo.setAttribute("color", new THREE.BufferAttribute(sc, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      size: 0.55, vertexColors: true, transparent: true, opacity: 0.85,
      depthWrite: false, sizeAttenuation: true,
    }));
    scene.add(stars);

    // nebulas
    const nebCols = ["70,90,200", "120,80,190", "60,140,180", "40,60,150"];
    const nebulas = [];
    for (let i = 0; i < 5; i++) {
      const spr = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture(nebCols[i % nebCols.length]),
        transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      const s = 90 + Math.random() * 120;
      spr.scale.set(s, s * (0.5 + Math.random() * 0.5), 1);
      spr.position.set((Math.random() - 0.5) * 180, (Math.random() - 0.5) * 70, -80 - Math.random() * 120);
      scene.add(spr);
      nebulas.push(spr);
    }

    const world = {
      scene, camera, renderer,
      planets: new Map(),
      camPos: camera.position.clone(),
      camTarget: new THREE.Vector3(0, 14, 44),
      look: new THREE.Vector3(0, 0, 0),
      lookTarget: new THREE.Vector3(0, 0, 0),
      blooms: [],
      stars, mouse: { x: 0, y: 0 },
      raycaster: new THREE.Raycaster(),
      clock: new THREE.Clock(),
      hovered: null,
      orbit: null, // {theta, alpha, dragging} — user-controlled rotation around focused planet
    };
    worldRef.current = world;

    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    let raf;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(world.clock.getDelta(), 0.05);
      const t = world.clock.elapsedTime;
      const motion = reducedMotion.current ? 0.25 : 1;

      stars.rotation.y += dt * 0.004 * motion;

      const v = viewRef.current;
      for (const [pid, rec] of world.planets) {
        const focused = v.mode === "planet" && v.projectId === pid;
        const target = focused ? 1 : 0;
        rec.explode += (target - rec.explode) * Math.min(1, dt * 2.4);
        rec.spinAngle += dt * rec.spinSpeed * motion * (1 - 0.75 * rec.explode);
        rec.spin.rotation.y = rec.spinAngle;
        const amt = rec.explode * (rec.R * 0.55 + 0.45);
        rec.chunks.forEach((ch) => {
          if (ch.isCore) {
            const pulse = 1 + Math.sin(t * 1.6) * 0.05 * motion;
            ch.mesh.scale.setScalar(pulse);
          } else {
            ch.mesh.position.copy(ch.dir).multiplyScalar(amt);
          }
        });
        if (rec.ring) rec.ring.rotation.z += dt * 0.02 * motion;
        rec.dust.forEach((d, i) => {
          if (d.anim) {
            d.anim.t = Math.min(1, d.anim.t + dt / 1.35);
            const k = 1 - d.anim.t;
            const ang = d.phase + t * d.speed + d.anim.t * 4.0;
            const rr = d.orbitR * k * k;
            d.cluster.position.set(
              Math.cos(ang) * rr,
              Math.sin(ang * 0.7 + d.incline) * rr * 0.35,
              Math.sin(ang) * rr
            );
            d.points.material.opacity = 0.95 * k + 0.05;
            d.cluster.scale.setScalar(Math.max(0.05, k));
          } else {
            const ang = d.phase + t * d.speed * motion;
            d.cluster.position.set(
              Math.cos(ang) * d.orbitR,
              Math.sin(ang * 0.7 + d.incline) * d.orbitR * 0.35,
              Math.sin(ang) * d.orbitR
            );
            const tw = 0.75 + Math.sin(t * 2 + d.phase * 7) * 0.2;
            d.points.material.opacity = tw;
          }
        });
      }

      // blooms
      for (let i = world.blooms.length - 1; i >= 0; i--) {
        const b = world.blooms[i];
        b.t += dt / b.dur;
        const e = 1 - Math.pow(1 - Math.min(b.t, 1), 3);
        const s = b.from + (b.to - b.from) * e;
        b.sprite.scale.set(s, s, 1);
        b.sprite.material.opacity = b.op * (1 - e);
        if (b.t >= 1) {
          scene.remove(b.sprite);
          b.sprite.material.map.dispose();
          b.sprite.material.dispose();
          world.blooms.splice(i, 1);
        }
      }

      // orbit around the focused planet (user-controlled via drag)
      if (v.mode === "planet" && world.orbit) {
        const rec = world.planets.get(v.projectId);
        if (rec) {
          const baseHeight = rec.R * 0.9;
          const baseDist = rec.R * 3.4 + 4.2;
          const radius = Math.hypot(baseHeight, baseDist);
          const alpha = world.orbit.alpha;
          const theta = world.orbit.theta;
          const horizontalDist = radius * Math.cos(alpha);
          const p = rec.group.position;
          world.camTarget.set(
            p.x + horizontalDist * Math.sin(theta),
            p.y + radius * Math.sin(alpha),
            p.z + horizontalDist * Math.cos(theta)
          );
          world.lookTarget.copy(p);
        }
      }

      // camera glide
      const drift = (v.mode === "galaxy" && !reducedMotion.current) ? 1 : 0;
      const px = world.mouse.x * 3 * drift, py = world.mouse.y * 1.6 * drift;
      if (v.mode === "planet" && world.orbit?.dragging) {
        world.camPos.set(world.camTarget.x, world.camTarget.y, world.camTarget.z);
      } else {
        world.camPos.lerp(new THREE.Vector3(
          world.camTarget.x + px, world.camTarget.y - py, world.camTarget.z
        ), Math.min(1, dt * 2.2));
      }
      camera.position.copy(world.camPos);
      world.look.lerp(world.lookTarget, Math.min(1, dt * 2.6));
      camera.lookAt(world.look);

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      for (const [, rec] of world.planets) disposeRecord(rec);
      starGeo.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      worldRef.current = null;
    };
  }, []);

  /* ---------- scene sync ---------- */
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    const keep = new Set(projects.map((p) => p.id));
    for (const [id, rec] of [...world.planets]) {
      if (!keep.has(id)) {
        world.scene.remove(rec.group);
        disposeRecord(rec);
        world.planets.delete(id);
      }
    }
    const positions = computeLayout(projects);
    let outer = 20;
    projects.forEach((p, i) => {
      const pos = positions[i];
      outer = Math.max(outer, pos.length() + radiusFor(p));
      const sig = signature(p);
      let rec = world.planets.get(p.id);
      if (rec && rec.sig !== sig) {
        const explode = rec.explode, spinAngle = rec.spinAngle;
        world.scene.remove(rec.group);
        disposeRecord(rec);
        rec = null;
        const fresh = buildPlanetRecord(p, pos);
        fresh.explode = explode;
        fresh.spinAngle = spinAngle;
        world.scene.add(fresh.group);
        world.planets.set(p.id, fresh);
      } else if (!rec) {
        const fresh = buildPlanetRecord(p, pos);
        world.scene.add(fresh.group);
        world.planets.set(p.id, fresh);
      } else {
        rec.group.position.copy(pos);
      }
    });
    world.galaxyExtent = outer;
  }, [projects]);

  /* ---------- camera targets (galaxy) ---------- */
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    if (view.mode === "galaxy") {
      const n = projects.length;
      const far = Math.max(30 + n * 6.5, (world.galaxyExtent || 20) * 1.7);
      world.camTarget.set(0, 12 + n * 1.5 + (world.galaxyExtent || 20) * 0.12, Math.min(420, far));
      world.lookTarget.set(0, 0, 0);
    }
  }, [view.mode, projects]);

  /* ---------- orbit init on planet focus (drag-to-rotate state) ---------- */
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    if (view.mode === "planet") {
      const rec = world.planets.get(view.projectId);
      const R = rec ? rec.R : 1.5;
      const baseHeight = R * 0.9;
      const baseDist = R * 3.4 + 4.2;
      world.orbit = { theta: 0, alpha: Math.atan2(baseHeight, baseDist), dragging: false };
    } else {
      world.orbit = null;
    }
  }, [view.mode, view.projectId]);

  /* ---------- pointer interaction ---------- */
  const pick = useCallback((clientX, clientY) => {
    const world = worldRef.current;
    const mount = mountRef.current;
    if (!world || !mount) return null;
    const rect = mount.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((clientY - rect.top) / rect.height) * 2 + 1;
    world.raycaster.setFromCamera({ x, y }, world.camera);
    const v = viewRef.current;
    if (v.mode === "galaxy") {
      const hits = [...world.planets.values()].map((r) => r.hit);
      const inter = world.raycaster.intersectObjects(hits, false);
      if (inter.length) return { type: "planet", projectId: inter[0].object.userData.projectId };
    } else {
      const rec = world.planets.get(v.projectId);
      if (!rec) return null;
      const objs = [];
      rec.chunks.forEach((c) => { if (!c.isCore && c.task) objs.push(c.mesh); });
      rec.dust.forEach((d) => { if (!d.anim) objs.push(d.clickSphere); });
      const inter = world.raycaster.intersectObjects(objs, false);
      if (inter.length) {
        const o = inter[0].object;
        if (o.userData.dustTaskId) return { type: "dust", taskId: o.userData.dustTaskId };
        if (o.userData.taskId) return { type: "chunk", taskId: o.userData.taskId };
      }
    }
    return null;
  }, []);

  const selectFromHit = useCallback((hit) => {
    if (!hit) { setSelected(null); return; }
    if (hit.type === "planet") {
      setView({ mode: "planet", projectId: hit.projectId });
      setSelected(null);
      setHoverTip(null);
    } else {
      setSelected({ projectId: viewRef.current.projectId, taskId: hit.taskId });
    }
  }, []);

  const onCanvasPointerDown = useCallback((e) => {
    const world = worldRef.current;
    const drag = dragRef.current;
    drag.down = true;
    drag.moved = false;
    drag.startX = e.clientX;
    drag.startY = e.clientY;
    if (world?.orbit) {
      drag.startTheta = world.orbit.theta;
      drag.startAlpha = world.orbit.alpha;
    }
    try { mountRef.current?.setPointerCapture?.(e.pointerId); } catch (err) { /* best-effort */ }
  }, []);

  const onCanvasPointerUp = useCallback((e) => {
    const world = worldRef.current;
    try { mountRef.current?.releasePointerCapture?.(e.pointerId); } catch (err) { /* best-effort */ }
    const drag = dragRef.current;
    drag.down = false;
    if (world?.orbit) world.orbit.dragging = false;
    if (!drag.moved) {
      selectFromHit(pick(e.clientX, e.clientY));
    }
  }, [pick, selectFromHit]);

  const ORBIT_SENSITIVITY = 0.006;
  const ORBIT_ALPHA_LIMIT = 1.45;

  const onCanvasMove = useCallback((e) => {
    const world = worldRef.current;
    const mount = mountRef.current;
    if (!world || !mount) return;
    const rect = mount.getBoundingClientRect();
    world.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    world.mouse.y = ((e.clientY - rect.top) / rect.height) * 2 - 1;

    const drag = dragRef.current;
    if (drag.down) {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) drag.moved = true;
      if (viewRef.current.mode === "planet" && world.orbit) {
        world.orbit.dragging = true;
        world.orbit.theta = drag.startTheta - dx * ORBIT_SENSITIVITY;
        world.orbit.alpha = clamp(drag.startAlpha + dy * ORBIT_SENSITIVITY, -ORBIT_ALPHA_LIMIT, ORBIT_ALPHA_LIMIT);
      }
      mount.style.cursor = "grabbing";
      setHoverTip(null);
      return;
    }

    const hit = pick(e.clientX, e.clientY);
    mount.style.cursor = hit ? "pointer" : (viewRef.current.mode === "planet" ? "grab" : "default");
    // hover glow on chunks
    for (const [, rec] of world.planets) {
      rec.chunks.forEach((c) => {
        if (!c.isCore) c.mesh.material.emissive.setHex(0x000000);
      });
    }
    if (hit && viewRef.current.mode === "planet" && hit.type === "chunk") {
      const rec = world.planets.get(viewRef.current.projectId);
      const c = rec?.chunks.find((ch) => ch.task && ch.task.id === hit.taskId);
      if (c) c.mesh.material.emissive.setHex(0x1c2f66);
    }
    if (hit && viewRef.current.mode === "galaxy") {
      const p = projectsRef.current.find((pp) => pp.id === hit.projectId);
      if (p) {
        setHoverTip({ x: e.clientX - rect.left, y: e.clientY - rect.top, label: p.name });
        return;
      }
    }
    setHoverTip(null);
  }, [pick]);

  /* ---------- bloom helper ---------- */
  const spawnBloom = useCallback((projectId, big) => {
    const world = worldRef.current;
    if (!world) return;
    const rec = world.planets.get(projectId);
    if (!rec) return;
    const rgb = `${Math.round(rec.pal.glow.r * 255)},${Math.round(rec.pal.glow.g * 255)},${Math.round(rec.pal.glow.b * 255)}`;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture(big ? "220,235,255" : rgb),
      transparent: true, opacity: big ? 0.9 : 0.7,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    spr.position.copy(rec.group.position);
    world.scene.add(spr);
    world.blooms.push({
      sprite: spr, t: 0, dur: big ? 1.8 : 1.1,
      from: rec.R, to: rec.R * (big ? 14 : 6), op: big ? 0.9 : 0.7,
    });
  }, []);

  /* ---------- data actions ---------- */
  const addProject = useCallback(() => {
    const name = fProjName.trim();
    if (!name) return;
    const p = { id: uid(), name, desc: fProjDesc.trim(), createdAt: Date.now(), tasks: [] };
    setProjects((ps) => [...ps, p]);
    setFProjName(""); setFProjDesc("");
    setIntakeMode("task");
    setFTaskProject(p.id);
  }, [fProjName, fProjDesc]);

  const addTask = useCallback(() => {
    const title = fTaskTitle.trim();
    if (!title || !fTaskProject) return;
    setProjects((ps) => ps.map((p) => p.id === fTaskProject
      ? { ...p, tasks: [...p.tasks, { id: uid(), title, notes: fTaskNotes.trim(), due: fTaskDue || null, done: false, createdAt: Date.now(), completedAt: null }] }
      : p));
    setFTaskTitle(""); setFTaskNotes(""); setFTaskDue("");
  }, [fTaskTitle, fTaskNotes, fTaskDue, fTaskProject]);

  const completeTask = useCallback((projectId, taskId) => {
    const world = worldRef.current;
    const rec = world?.planets.get(projectId);
    const d = rec?.dust.find((dd) => dd.task.id === taskId);
    const finish = () => {
      let projectDone = false;
      setProjects((ps) => ps.map((p) => {
        if (p.id !== projectId) return p;
        const tasks = p.tasks.map((t) => t.id === taskId ? { ...t, done: true, completedAt: Date.now() } : t);
        projectDone = tasks.length > 0 && tasks.every((t) => t.done);
        return { ...p, tasks };
      }));
      setTimeout(() => {
        spawnBloom(projectId, projectDone);
        chime(projectDone);
      }, 60);
    };
    setSelected(null);
    if (d && !reducedMotion.current) {
      d.anim = { t: 0 };
      setTimeout(finish, 1350);
    } else {
      finish();
    }
  }, [spawnBloom, chime]);

  const uncompleteTask = useCallback((projectId, taskId) => {
    setSelected(null);
    setProjects((ps) => ps.map((p) => p.id === projectId
      ? { ...p, tasks: p.tasks.map((t) => t.id === taskId ? { ...t, done: false, completedAt: null } : t) }
      : p));
  }, []);

  const deleteTask = useCallback((projectId, taskId) => {
    setSelected(null);
    setProjects((ps) => ps.map((p) => p.id === projectId
      ? { ...p, tasks: p.tasks.filter((t) => t.id !== taskId) }
      : p));
  }, []);

  const deleteProject = useCallback((projectId) => {
    setConfirmDelete(null);
    setSelected(null);
    setView({ mode: "galaxy", projectId: null });
    setProjects((ps) => ps.filter((p) => p.id !== projectId));
  }, []);

  /* ---------- derived ---------- */
  const focusProject = view.mode === "planet" ? projects.find((p) => p.id === view.projectId) : null;
  const selTask = selected
    ? projects.find((p) => p.id === selected.projectId)?.tasks.find((t) => t.id === selected.taskId)
    : null;
  const overdue = (t) => t.due && !t.done && new Date(t.due + "T23:59:59") < new Date();

  /* ---------- render ---------- */
  return (
    <div className="orrery-root">
      <style>{css}</style>
      <div
        ref={mountRef}
        className="canvas-mount"
        onPointerDown={onCanvasPointerDown}
        onPointerMove={onCanvasMove}
        onPointerUp={onCanvasPointerUp}
      />
      {hoverTip && (
        <div className="tip" style={{ left: hoverTip.x + 14, top: hoverTip.y - 10 }}>
          {hoverTip.label}
        </div>
      )}

      {/* header */}
      <header className="hud top-left">
        {view.mode === "planet" ? (
          <button className="ghost-btn" onClick={() => { setView({ mode: "galaxy", projectId: null }); setSelected(null); }}>
            ← Galaxy
          </button>
        ) : (
          <div className="brand">
            <div className="brand-name">ORRERY</div>
            <div className="brand-sub">a galaxy built from your work</div>
          </div>
        )}
      </header>

      {/* focused project header */}
      {focusProject && (
        <div className="hud project-head">
          <div className="ph-name">{focusProject.name}</div>
          {focusProject.desc ? <div className="ph-desc">{focusProject.desc}</div> : null}
          <div className="ph-meta">
            {focusProject.tasks.filter((t) => t.done).length} of {focusProject.tasks.length} crystallized
            {focusProject.tasks.length > 0 && focusProject.tasks.every((t) => t.done) && (
              <span className="complete-tag"> · world complete</span>
            )}
          </div>
          <div className="bar">
            <div className="bar-fill" style={{
              width: focusProject.tasks.length
                ? `${(focusProject.tasks.filter((t) => t.done).length / focusProject.tasks.length) * 100}%`
                : "0%",
            }} />
          </div>
          <div className="ph-actions">
            <button className="ghost-btn sm" onClick={() => {
              setIntakeMode("task"); setFTaskProject(focusProject.id); setIntakeOpen(true);
            }}>+ Add task</button>
            {confirmDelete === focusProject.id ? (
              <span className="confirm">
                Delete this world and all its tasks?
                <button className="danger-btn sm" onClick={() => deleteProject(focusProject.id)}>Delete</button>
                <button className="ghost-btn sm" onClick={() => setConfirmDelete(null)}>Keep</button>
              </span>
            ) : (
              <button className="ghost-btn sm dim" onClick={() => setConfirmDelete(focusProject.id)}>Delete world</button>
            )}
          </div>
          <div className="ph-hint">
            {focusProject.tasks.length === 0
              ? "A newborn core. Add tasks to give it matter."
              : "Tap a fragment to recall its work · tap orbiting stardust to view open tasks · drag to rotate"}
          </div>
        </div>
      )}

      {/* task detail */}
      {selTask && (
        <aside className="hud detail">
          <div className="d-status">{selTask.done ? "◆ Crystallized" : "◇ In orbit"}</div>
          <div className="d-title">{selTask.title}</div>
          {selTask.notes ? <div className="d-notes">{selTask.notes}</div> : null}
          <div className="d-rows">
            {selTask.due && (
              <div className={"d-row" + (overdue(selTask) ? " warn" : "")}>
                Due {fmtDate(selTask.due)}{overdue(selTask) ? " · past due" : ""}
              </div>
            )}
            {selTask.completedAt && (
              <div className="d-row">Completed {new Date(selTask.completedAt).toLocaleDateString()}</div>
            )}
          </div>
          <div className="d-actions">
            {!selTask.done ? (
              <button className="primary-btn" onClick={() => completeTask(selected.projectId, selTask.id)}>
                ✦ Crystallize
              </button>
            ) : (
              <button className="ghost-btn" onClick={() => uncompleteTask(selected.projectId, selTask.id)}>
                Return to stardust
              </button>
            )}
            {confirmDeleteTask === selTask.id ? (
              <span className="confirm">
                Delete this task?
                <button className="danger-btn sm" onClick={() => deleteTask(selected.projectId, selTask.id)}>Delete</button>
                <button className="ghost-btn sm" onClick={() => setConfirmDeleteTask(null)}>Keep</button>
              </span>
            ) : (
              <button className="ghost-btn dim" onClick={() => setConfirmDeleteTask(selTask.id)}>Delete</button>
            )}
            <button className="ghost-btn dim" onClick={() => setSelected(null)}>Close</button>
          </div>
        </aside>
      )}

      {/* empty state */}
      {loaded && projects.length === 0 && (
        <div className="hud empty">
          <div className="empty-title">Your galaxy is empty.</div>
          <div className="empty-sub">Every project becomes a world. Every finished task becomes part of it.</div>
          <button className="primary-btn" onClick={() => { setIntakeMode("project"); setIntakeOpen(true); }}>
            Begin your first world
          </button>
        </div>
      )}

      {/* index */}
      <div className="hud index-wrap">
        <button className="ghost-btn sm" onClick={() => setIndexOpen((o) => !o)}>
          {indexOpen ? "Close index" : `Index · ${projects.length}`}
        </button>
        {indexOpen && (
          <div className="index">
            {projects.length === 0 && <div className="idx-empty">No worlds yet</div>}
            {projects.map((p) => {
              const done = p.tasks.filter((t) => t.done).length;
              return (
                <button key={p.id} className="idx-row" onClick={() => {
                  setView({ mode: "planet", projectId: p.id });
                  setSelected(null); setIndexOpen(false);
                }}>
                  <span className="idx-name">{p.name}</span>
                  <span className="idx-meta">{done}/{p.tasks.length}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* bottom controls */}
      <div className="hud bottom-right">
        {saveNote && <span className="save-note">{saveNote}</span>}
        <button className="ghost-btn" onClick={cycleSound} aria-label="Cycle ambient sound mode">
          {soundMode === "off" ? "♪ sound off" : soundMode === "ambient" ? "♪ ambient hum" : "♪ soft strings"}
        </button>
        <button className="primary-btn" onClick={() => {
          setIntakeMode(projects.length ? "task" : "project");
          if (projects.length && !fTaskProject) setFTaskProject(view.projectId || projects[0].id);
          setIntakeOpen(true);
        }}>
          + Intake
        </button>
      </div>

      {/* intake modal */}
      {intakeOpen && (
        <div className="scrim" onClick={(e) => { if (e.target === e.currentTarget) setIntakeOpen(false); }}>
          <div className="modal" role="dialog" aria-label="Intake">
            <div className="m-tabs">
              <button className={"m-tab" + (intakeMode === "project" ? " on" : "")} onClick={() => setIntakeMode("project")}>
                New world
              </button>
              <button
                className={"m-tab" + (intakeMode === "task" ? " on" : "")}
                onClick={() => setIntakeMode("task")}
                disabled={projects.length === 0}
              >
                New task
              </button>
            </div>
            {intakeMode === "project" ? (
              <div className="m-body">
                <label className="f-label" htmlFor="pname">Project name</label>
                <input id="pname" className="f-input" value={fProjName} onChange={(e) => setFProjName(e.target.value)}
                  placeholder="e.g. Q3 client migration" autoFocus />
                <label className="f-label" htmlFor="pdesc">Description (optional)</label>
                <textarea id="pdesc" className="f-input" rows={2} value={fProjDesc} onChange={(e) => setFProjDesc(e.target.value)}
                  placeholder="What is this world?" />
                <div className="m-actions">
                  <button className="primary-btn" disabled={!fProjName.trim()} onClick={addProject}>
                    Birth this world
                  </button>
                  <button className="ghost-btn" onClick={() => setIntakeOpen(false)}>Close</button>
                </div>
                <div className="m-hint">A new world begins as a molten core with no matter — its tasks will build it.</div>
              </div>
            ) : (
              <div className="m-body">
                <label className="f-label" htmlFor="tproj">World</label>
                <select id="tproj" className="f-input" value={fTaskProject} onChange={(e) => setFTaskProject(e.target.value)}>
                  <option value="" disabled>Select a world…</option>
                  {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <label className="f-label" htmlFor="ttitle">Task title</label>
                <input id="ttitle" className="f-input" value={fTaskTitle} onChange={(e) => setFTaskTitle(e.target.value)}
                  placeholder="e.g. Draft the kickoff email" autoFocus />
                <label className="f-label" htmlFor="tnotes">Notes (optional)</label>
                <textarea id="tnotes" className="f-input" rows={2} value={fTaskNotes} onChange={(e) => setFTaskNotes(e.target.value)}
                  placeholder="Details worth remembering" />
                <label className="f-label" htmlFor="tdue">Due date (optional)</label>
                <input id="tdue" type="date" className="f-input" value={fTaskDue} onChange={(e) => setFTaskDue(e.target.value)} />
                <div className="m-actions">
                  <button className="primary-btn" disabled={!fTaskTitle.trim() || !fTaskProject} onClick={addTask}>
                    Send into orbit
                  </button>
                  <button className="ghost-btn" onClick={() => setIntakeOpen(false)}>Close</button>
                </div>
                <div className="m-hint">New tasks orbit their world as stardust until you crystallize them.</div>
              </div>
            )}
          </div>
        </div>
      )}

      {!loaded && <div className="hud loading">waking the galaxy…</div>}
    </div>
  );
}

/* ---------------- styles ---------------- */
const css = `
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Outfit:wght@300;400;500&display=swap');

.orrery-root {
  --ink: #e9edff;
  --ink-dim: #9aa6cf;
  --panel: rgba(9, 14, 34, 0.74);
  --line: rgba(150, 175, 255, 0.16);
  --accent: #9db8ff;
  --warn: #e8c58a;
  position: fixed; inset: 0; overflow: hidden;
  background: #04060f;
  font-family: 'Outfit', system-ui, sans-serif;
  color: var(--ink);
  font-size: 15px;
}
.canvas-mount { position: absolute; inset: 0; }
.hud { position: absolute; z-index: 5; }
.tip {
  position: absolute; z-index: 6; pointer-events: none;
  font-family: 'Cormorant Garamond', serif; font-size: 19px; font-style: italic;
  color: var(--ink); text-shadow: 0 0 12px rgba(120,150,255,0.8);
}
.top-left { top: 18px; left: 18px; }
.brand-name {
  font-family: 'Cormorant Garamond', serif; font-weight: 600;
  font-size: 26px; letter-spacing: 0.34em; color: var(--ink);
}
.brand-sub { font-size: 12px; color: var(--ink-dim); letter-spacing: 0.08em; margin-top: 2px; }

.project-head {
  top: 18px; left: 50%; transform: translateX(-50%);
  width: min(560px, 92vw); text-align: center;
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 16px; padding: 14px 18px 12px;
  backdrop-filter: blur(10px);
}
.ph-name { font-family: 'Cormorant Garamond', serif; font-size: 26px; font-weight: 600; }
.ph-desc { color: var(--ink-dim); font-size: 13px; margin-top: 2px; }
.ph-meta { font-size: 13px; color: var(--ink-dim); margin-top: 6px; }
.complete-tag { color: var(--accent); }
.bar { height: 3px; background: rgba(150,175,255,0.14); border-radius: 2px; margin: 8px auto 0; width: 70%; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #7fa0ff, #b7e6ff); border-radius: 2px; transition: width 0.6s ease; }
.ph-actions { display: flex; gap: 8px; justify-content: center; align-items: center; margin-top: 10px; flex-wrap: wrap; }
.ph-hint { font-size: 11.5px; color: var(--ink-dim); margin-top: 8px; opacity: 0.85; }
.confirm { font-size: 12.5px; color: var(--warn); display: inline-flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: center; }

.detail {
  right: 18px; top: 50%; transform: translateY(-50%);
  width: min(320px, 88vw);
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 16px; padding: 18px; backdrop-filter: blur(10px);
}
.d-status { font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--accent); }
.d-title { font-family: 'Cormorant Garamond', serif; font-size: 24px; font-weight: 600; margin-top: 6px; line-height: 1.2; }
.d-notes { color: var(--ink-dim); font-size: 13.5px; margin-top: 8px; white-space: pre-wrap; line-height: 1.5; }
.d-rows { margin-top: 10px; }
.d-row { font-size: 12.5px; color: var(--ink-dim); margin-top: 2px; }
.d-row.warn { color: var(--warn); }
.d-actions { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }

.empty {
  left: 50%; top: 50%; transform: translate(-50%, -50%);
  text-align: center; max-width: 92vw;
}
.empty-title { font-family: 'Cormorant Garamond', serif; font-size: 34px; font-weight: 500; }
.empty-sub { color: var(--ink-dim); margin: 8px 0 18px; font-size: 14px; }

.index-wrap { left: 18px; bottom: 18px; display: flex; flex-direction: column-reverse; align-items: flex-start; gap: 8px; }
.index {
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 8px; width: min(260px, 80vw); max-height: 44vh; overflow: auto;
  backdrop-filter: blur(10px);
}
.idx-row {
  display: flex; justify-content: space-between; gap: 12px; width: 100%;
  background: none; border: none; color: var(--ink); cursor: pointer;
  padding: 9px 10px; border-radius: 9px; font-size: 14px; text-align: left;
  font-family: inherit;
}
.idx-row:hover { background: rgba(150,175,255,0.10); }
.idx-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.idx-meta { color: var(--ink-dim); font-size: 12px; flex-shrink: 0; }
.idx-empty { color: var(--ink-dim); font-size: 13px; padding: 8px 10px; }

.bottom-right { right: 18px; bottom: 18px; display: flex; gap: 10px; align-items: center; }
.save-note { font-size: 11.5px; color: var(--ink-dim); }

.primary-btn {
  background: linear-gradient(180deg, rgba(130,160,255,0.28), rgba(90,120,220,0.20));
  border: 1px solid rgba(160,185,255,0.45); color: var(--ink);
  border-radius: 999px; padding: 11px 20px; font-size: 14px; cursor: pointer;
  font-family: inherit; letter-spacing: 0.02em;
  box-shadow: 0 0 22px rgba(120,150,255,0.18);
  transition: box-shadow 0.2s ease, background 0.2s ease;
}
.primary-btn:hover { box-shadow: 0 0 30px rgba(140,170,255,0.35); }
.primary-btn:disabled { opacity: 0.45; cursor: default; box-shadow: none; }
.ghost-btn {
  background: rgba(12,18,40,0.6); border: 1px solid var(--line); color: var(--ink);
  border-radius: 999px; padding: 10px 16px; font-size: 13.5px; cursor: pointer;
  font-family: inherit; backdrop-filter: blur(8px);
  transition: border-color 0.2s ease;
}
.ghost-btn:hover { border-color: rgba(170,195,255,0.4); }
.ghost-btn.sm { padding: 7px 13px; font-size: 12.5px; }
.ghost-btn.dim { color: var(--ink-dim); }
.ghost-btn:disabled { opacity: 0.4; cursor: default; }
.danger-btn {
  background: rgba(120,60,60,0.3); border: 1px solid rgba(230,150,140,0.4);
  color: #f2c9c2; border-radius: 999px; cursor: pointer; font-family: inherit;
}
.danger-btn.sm { padding: 7px 13px; font-size: 12.5px; }

.scrim {
  position: fixed; inset: 0; z-index: 20;
  background: rgba(2,4,12,0.62); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center; padding: 18px;
}
.modal {
  width: min(430px, 94vw); background: rgba(10,15,36,0.92);
  border: 1px solid var(--line); border-radius: 18px; padding: 18px;
  box-shadow: 0 20px 80px rgba(0,0,0,0.5), 0 0 60px rgba(90,120,255,0.08);
  max-height: 86vh; overflow: auto;
}
.m-tabs { display: flex; gap: 8px; margin-bottom: 14px; }
.m-tab {
  flex: 1; background: none; border: 1px solid var(--line); color: var(--ink-dim);
  border-radius: 10px; padding: 10px; cursor: pointer; font-family: inherit; font-size: 14px;
}
.m-tab.on { color: var(--ink); border-color: rgba(160,185,255,0.5); background: rgba(130,160,255,0.10); }
.m-tab:disabled { opacity: 0.35; cursor: default; }
.f-label { display: block; font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-dim); margin: 12px 0 5px; }
.f-input {
  width: 100%; box-sizing: border-box; background: rgba(6,10,26,0.8);
  border: 1px solid var(--line); border-radius: 10px; color: var(--ink);
  padding: 11px 12px; font-size: 14.5px; font-family: inherit; outline: none;
  resize: vertical;
}
.f-input:focus { border-color: rgba(170,195,255,0.55); box-shadow: 0 0 0 3px rgba(130,160,255,0.12); }
.m-actions { display: flex; gap: 10px; margin-top: 16px; }
.m-hint { font-size: 12px; color: var(--ink-dim); margin-top: 12px; line-height: 1.5; }

.loading { left: 50%; bottom: 26px; transform: translateX(-50%); color: var(--ink-dim); font-size: 13px; }

button:focus-visible, .f-input:focus-visible { outline: 2px solid rgba(170,195,255,0.7); outline-offset: 2px; }

@media (max-width: 640px) {
  .detail { top: auto; bottom: 84px; transform: none; right: 50%; transform: translateX(50%); }
  .project-head { padding: 10px 12px; }
  .ph-name { font-size: 21px; }
}
@media (prefers-reduced-motion: reduce) {
  .bar-fill { transition: none; }
}
`;
