import React, { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import * as Tone from "tone";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { supabase, loadCloudProjects, saveCloudProjects } from "./supabaseClient.js";

// Injected by .github/workflows/deploy.yml at build time (VITE_-prefixed env
// vars are auto-exposed by Vite); falls back to "dev" for local `npm run dev`.
const BUILD_SHA = (import.meta.env.VITE_BUILD_SHA || "dev").slice(0, 7);
const BUILD_TIME_RAW = import.meta.env.VITE_BUILD_TIME || "";
const BUILD_TIME_LABEL = BUILD_TIME_RAW
  ? new Date(BUILD_TIME_RAW).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
  : "local";

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
// A planet's visual footprint reaches well past its body: the additive
// atmosphere glow's meaningfully-bright zone runs to ~2.3R (it fades further
// past that), and Q4 "drifting" dust orbits out to ~1.3x the normal gap.
// Spacing needs to clear THIS, not the small click hit-sphere, or halos
// wash into each other at large sizes.
const PLANET_VISUAL_SPREAD = 2.6;

// The central star every project-planet orbits. One system for now; a later
// pass may split projects by year into their own solar systems under a wider
// galaxy view, at which point this becomes per-system rather than singular.
// Kept comfortably above radiusFor's max (16) so the star always reads as the
// biggest body in the scene — a planet that outgrows its own sun breaks the
// scale hierarchy that makes this look like a solar system at all.
const STAR_RADIUS = 24;
// Scales with STAR_RADIUS, so a bigger star automatically pushes every
// planet's orbit further out too — "ample space from the sun" falls out of
// the same knob rather than needing a separate constant.
const STAR_CLEARANCE = STAR_RADIUS * PLANET_VISUAL_SPREAD + 10;

// Inner planets revolve faster than outer ones (loosely Keplerian — period
// grows with radius) but everything is slowed to a calm, ambient drift rather
// than anything physically real: a full revolution takes many minutes even
// for the innermost planet.
const ORBIT_ANGULAR_K = 0.1;
function orbitSpeedFor(radius) {
  return ORBIT_ANGULAR_K / Math.sqrt(radius);
}

// Scroll/pinch-to-zoom clamps — a multiplier on the existing camera-distance
// formula for each view mode, not a separate camera system. Trackpad pinch
// arrives as wheel events with ctrlKey set and much larger deltaY, hence the
// separate (smaller) sensitivity for that case.
const PLANET_ZOOM_MIN = 0.5;
const PLANET_ZOOM_MAX = 2.2;
const GALAXY_ZOOM_MIN = 0.55;
const GALAXY_ZOOM_MAX = 1.8;
const ZOOM_WHEEL_SENSITIVITY = 0.0016;
const ZOOM_PINCH_SENSITIVITY = 0.012;

// Returns orbit descriptors {radius, y, phase} per project — the star-relative
// circular path each planet's group.position is driven from every frame (see
// the animate loop), rather than a one-time static position.
// Sunflower/Fibonacci-disc packing: radius grows with sqrt of cumulative
// footprint AREA rather than a linear cumulative sum of radii. For N
// similar-sized planets that's O(sqrt(N)) instead of O(N) — at 100+ projects
// the old linear sum pushed galaxyExtent into the thousands of units, far
// past any distance the camera could actually reach. The golden-angle phase
// spacing (ga) already gives good angular distribution and is unchanged.
const GALAXY_PACK_K = 2.4;
function computeLayout(projects) {
  const ga = 2.39996;
  const orbits = [];
  let area = 0;
  for (let i = 0; i < projects.length; i++) {
    const R = radiusFor(projects[i]);
    const footprint = R * PLANET_VISUAL_SPREAD + 4; // local footprint incl. breathing room
    area += footprint * footprint;
    const orbitR = STAR_CLEARANCE + GALAXY_PACK_K * Math.sqrt(area);
    const y = Math.sin(i * 0.27 + 0.6) * (2.6 + R * 0.35);
    const phase = i * ga;
    orbits.push({ radius: orbitR, y, phase });
  }
  return orbits;
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
  // Size = scope × completion. The old raw-done-count power curve made a
  // 90%-done small project render SMALLER than a 20%-done monster — size
  // read as "most accomplished-looking" when users scan for "closest to
  // done". Scope (total tasks, log-damped) still matters, but completion
  // ratio carries most of the growth, so finishing work is what makes a
  // world visibly swell. Honest side effect: adding open tasks shrinks a
  // planet slightly — the ratio dropped, there's more world left to build.
  // ratio carries 75% of the growth range: at these weights a 90%-done
  // 10-task world (R≈5.9) edges out a 20%-done 200-task one (R≈5.7) —
  // the exact inversion the old formula got wrong — while a completed
  // large world still out-sizes a completed small one.
  const ratio = total ? done / total : 0;
  const scope = Math.log1p(total);
  const growth = scope * 2.2 * (0.25 + 0.75 * ratio);
  return 1.0 + Math.min(15, growth);
}
function signature(p) {
  return p.id + "|" + p.tasks.map((t) => t.id + (t.done ? "1" : "0")).join(",");
}

/* ---------------- Eisenhower triage ----------------
 * Quadrants live on TASKS (a maintenance world can host one fire-drill task
 * without mislabeling the whole planet); planets aggregate their open tasks.
 * Urgency is DERIVED from the due date, never asked — it's a fact about time,
 * and a manually-set flag goes stale. Importance is the one human judgment,
 * asked once at intake (defaulting from the project's archetype) and
 * editable from the task detail panel. Absent `important` means true:
 * existing saved tasks were presumably worth tracking, and the toggle
 * exists to demote, not to nag. */
const URGENT_WINDOW_DAYS = 3;
function taskImportant(t) {
  return t.important !== false;
}
function taskUrgent(t) {
  if (!t.due || t.done) return false;
  // same end-of-day convention as overdue(): a task due today is urgent all day
  const due = new Date(t.due + "T23:59:59");
  const horizon = new Date(Date.now() + URGENT_WINDOW_DAYS * 24 * 60 * 60 * 1000);
  return due <= horizon;
}
// Overdue is a SUB-tier of urgent (every overdue task is already urgent —
// due < now implies due <= now+window), used only to escalate the visual
// treatment within q1/q3, not to create a fifth quadrant.
function taskOverdue(t) {
  return !!(t.due && !t.done && new Date(t.due + "T23:59:59") < new Date());
}
function taskQuadrant(t) {
  return taskImportant(t)
    ? (taskUrgent(t) ? "q1" : "q2")
    : (taskUrgent(t) ? "q3" : "q4");
}
function projectPriority(p) {
  let q1 = 0, q3 = 0;
  for (const t of p.tasks) {
    if (t.done) continue;
    const q = taskQuadrant(t);
    if (q === "q1") q1++;
    else if (q === "q3") q3++;
  }
  return { q1, q3 };
}
// Staleness — a "worth revisiting" signal, distinct from urgency. Adding a
// task counts as activity (not just finishing one), so a maintenance world
// someone is actively populating doesn't read as neglected before its first
// completion. createdAt/completedAt were always stamped but never shown.
const STALE_DAYS = 21;
function projectStaleDays(p) {
  let last = p.createdAt || Date.now();
  for (const t of p.tasks) {
    if (t.createdAt) last = Math.max(last, t.createdAt);
    if (t.completedAt) last = Math.max(last, t.completedAt);
  }
  return Math.floor((Date.now() - last) / 86400000);
}
function isProjectStale(p) {
  return projectStaleDays(p) >= STALE_DAYS;
}
// Unified task search — the single surface for "find a task without knowing
// which world it's on": real DOM rows, so it works for keyboard/screen-reader
// users the same as mouse, unlike 3D picking. Matches task title or parent
// world name (typing a world's name surfaces all its tasks too). Capped, not
// silently truncated — the caller shows "N more matches" when total > limit.
const TASK_SEARCH_LIMIT = 50;
function searchTasks(projects, query, limit = TASK_SEARCH_LIMIT) {
  const q = query.trim().toLowerCase();
  if (!q) return { results: [], total: 0 };
  const hits = [];
  for (const p of projects) {
    const nameMatch = p.name.toLowerCase().includes(q);
    for (const t of p.tasks) {
      if (nameMatch || t.title.toLowerCase().includes(q)) hits.push({ p, t });
    }
  }
  const rank = (t) => t.done ? 4 : { q1: 0, q3: 1, q2: 2, q4: 3 }[taskQuadrant(t)];
  hits.sort((a, b) => rank(a.t) - rank(b.t));
  return { results: hits.slice(0, limit), total: hits.length };
}
// Dust agitation — open-task stardust moves by triage state: Q1 orbits fast
// and tight (agitated, insistent), Q3 quick, Q2 calm baseline, Q4 slow and
// distant (drifting away). Motion is pre-attentive, so an agitated planet
// pops even peripherally; the ▲ labels/badges remain the static fallback.
// Speed applies LIVE (re-triage updates it seamlessly under accumulated
// angles); orbit distance is build-time only, so nothing teleports.
const DUST_QUAD_SPEED = { q1: 2.2, q2: 1.0, q3: 1.5, q4: 0.55 };
const DUST_QUAD_GAP = { q1: 0.8, q2: 1.0, q3: 1.0, q4: 1.3 };
// Dust severity color + size — within a focused planet, hue is unspent (the
// "hue = project identity" budget only binds in galaxy view, where color
// tells worlds apart; here every cluster shares one project), so it's the
// right channel for "click this one first". Never the SOLE channel: speed,
// orbit distance, and the ▲/△ glyph on the task label carry the same rank.
const DUST_QUAD_SIZE = { q1: 0.32, q2: 0.24, q3: 0.28, q4: 0.20 };
const DUST_QUAD_GLYPH = { q1: "▲ ", q2: "", q3: "△ ", q4: "" };
// Overdue escalation — a sub-tier WITHIN q1/q3 (q2/q4 are never urgent, so
// never overdue). "Due in 3 days" and "overdue by a week" otherwise render
// identically; this is the extra notch of color/size/speed/glyph that
// answers "which do I click first" once several q1 tasks are competing.
// Shape escalates too (▲!/△! vs ▲ /△ ), not just color, per the same
// color-blind guardrail as the base quadrant colors.
const OVERDUE_SPEED_BOOST = 1.3;
const DUST_QUAD_SIZE_OVERDUE = { q1: 0.38, q3: 0.32 };
const DUST_QUAD_GLYPH_OVERDUE = { q1: "▲! ", q3: "△! " };
function dustQuadColor(quad, pal, overdue) {
  if (overdue) {
    if (quad === "q1") return new THREE.Color(0xff5a3c); // crimson — do first, already late
    if (quad === "q3") return new THREE.Color(0xd98f3c); // burnt orange — overdue, not important
  }
  switch (quad) {
    case "q1": return new THREE.Color(0xffab66); // hot amber — do first
    case "q3": return new THREE.Color(0xe8dc8a); // pale caution yellow — due soon, not important
    case "q4": return new THREE.Color(0x5a6478); // dim slate — drifting
    default: return pal.accent.clone();          // q2 keeps the project's identity accent
  }
}
function dustQuadSize(quad, overdue) {
  return (overdue && DUST_QUAD_SIZE_OVERDUE[quad]) || DUST_QUAD_SIZE[quad];
}
function dustQuadGlyph(quad, overdue) {
  return (overdue && DUST_QUAD_GLYPH_OVERDUE[quad]) || DUST_QUAD_GLYPH[quad];
}
function dustQuadSpeed(quad, overdue) {
  return DUST_QUAD_SPEED[quad] * (overdue && (quad === "q1" || quad === "q3") ? OVERDUE_SPEED_BOOST : 1);
}
// Project archetypes — the user's five real work categories. This slice they
// only drive intake pre-fills (default importance, and exec fire drills
// pre-filling due = today so capture stays title + Enter); later slices may
// hang dust behavior or triage ordering off them.
const ARCHETYPES = [
  { key: "general", label: "General", importantDefault: true, dueToday: false },
  { key: "exec", label: "Executive Ad-Hoc", importantDefault: true, dueToday: true },
  { key: "flagship", label: "Flagship Deliverable", importantDefault: true, dueToday: false },
  // individual routine maintenance tasks are the classic "the program matters,
  // each task doesn't" case — overdue ones read Q3 (contain), not false-alarm
  // Q1; one tap promotes exceptions.
  { key: "maintenance", label: "Database Maintenance", importantDefault: false, dueToday: false },
  { key: "sprint", label: "Application Sprint", importantDefault: true, dueToday: false },
  { key: "governance", label: "Governance & Compliance", importantDefault: true, dueToday: false },
];
function archetypeFor(project) {
  return ARCHETYPES.find((a) => a.key === project?.archetype) || ARCHETYPES[0];
}
function todayISO(offsetDays = 0) {
  const d = new Date(Date.now() + offsetDays * 24 * 60 * 60 * 1000);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
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

/* ---------------- planet construction ----------------
 * Split into a cheap "shell" (always built, for every project, even at
 * 100+ concurrent planets) and an expensive "detail" layer (full per-vertex
 * terrain + task-chunk bucketing + stardust) that's only ever attached to
 * whichever single project is currently focused. The shell's low-poly
 * impostor mesh is what galaxy view actually renders for everyone else. */
function buildPlanetShell(project, orbit) {
  const pal = paletteFor(project);
  const rng = mulberry32(strHash(project.id));
  const R = radiusFor(project);
  const doneTasks = project.tasks.filter((t) => t.done);
  const group = new THREE.Group();
  group.position.set(
    Math.cos(orbit.phase) * orbit.radius,
    orbit.y,
    Math.sin(orbit.phase) * orbit.radius
  );
  const spin = new THREE.Group();
  group.add(spin);

  // cheap galaxy-view impostor — a low-poly solid-color sphere standing in
  // for the full terrain mesh. Hidden (not removed) while detail is attached.
  const impostorR = doneTasks.length ? R : 0.85;
  const impostorGeo = new THREE.IcosahedronGeometry(impostorR, 2);
  const impostorMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setHSL(pal.hue, pal.sat, 0.4),
    emissive: pal.glow, emissiveIntensity: doneTasks.length ? 0.22 : 0.5,
    roughness: 0.7, metalness: 0.08,
  });
  const impostor = new THREE.Mesh(impostorGeo, impostorMat);
  spin.add(impostor);

  // atmosphere glow
  const atmo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture(`${Math.round(pal.glow.r * 255)},${Math.round(pal.glow.g * 255)},${Math.round(pal.glow.b * 255)}`),
    transparent: true, opacity: doneTasks.length ? 0.5 : 0.32,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  const aScale = (doneTasks.length ? R : 0.9) * 4.6;
  atmo.scale.set(aScale, aScale, 1);
  group.add(atmo);

  // invisible galaxy-level click sphere
  const hit = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(R, 1.1) + 0.8, 10, 10),
    new THREE.MeshBasicMaterial({ visible: false })
  );
  hit.userData.projectId = project.id;
  group.add(hit);

  return {
    group, spin, impostor, chunks: [], dust: [], atmo, hit,
    R, pal, sig: signature(project), detailBuilt: false,
    explode: 0, spinAngle: Math.random() * Math.PI * 2,
    spinSpeed: 0.05 + rng() * 0.05,
    orbitRadius: orbit.radius, orbitY: orbit.y, orbitAngle: orbit.phase,
    orbitSpeed: orbitSpeedFor(orbit.radius),
  };
}

function attachPlanetDetail(rec, project) {
  if (rec.detailBuilt) return;
  const pal = rec.pal;
  const rng = mulberry32(strHash(project.id));
  const R = rec.R;
  const doneTasks = project.tasks.filter((t) => t.done);
  const openTasks = project.tasks.filter((t) => !t.done);
  const spin = rec.spin;
  const group = rec.group;
  const chunks = rec.chunks;

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

  // (rings retired: they were a per-project 50% coin flip — decoration that
  // read as a milestone but carried no information, teaching users to hunt
  // for a pattern that didn't exist. If a circular element returns, it
  // should encode something real, e.g. a completion arc.)

  // stardust clusters for open tasks — soft round cloud-puffs, not hard squares.
  // The glow texture and per-vertex jitter are NEUTRAL (grayscale); the actual
  // hue lives on material.color so severity recoloring on live re-triage is a
  // single material property write, not a vertex-buffer rewrite.
  const dust = rec.dust;
  const dustMap = glowTexture("255,255,255");
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
      colArr[k * 3] = jitter;
      colArr[k * 3 + 1] = jitter;
      colArr[k * 3 + 2] = jitter;
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    pg.setAttribute("color", new THREE.Float32BufferAttribute(colArr, 3));
    const quad = taskQuadrant(task);
    const overdue = taskOverdue(task);
    const pm = new THREE.PointsMaterial({
      map: dustMap, vertexColors: true, size: dustQuadSize(quad, overdue),
      color: dustQuadColor(quad, pal, overdue), transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });
    const points = new THREE.Points(pg, pm);
    const clickSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.5 + R * 0.06, 8, 8), // scales with R — a fixed radius became a tiny, hard-to-hit target on large planets
      new THREE.MeshBasicMaterial({ visible: false })
    );
    clickSphere.userData.dustTaskId = task.id;
    const cluster = new THREE.Group();
    cluster.add(points, clickSphere);
    group.add(cluster);
    // gap and per-task stagger both scale with R — a fixed offset (the old
    // "R + 1.35 + i*0.55") was only ~8% of R on a large planet, so dust
    // clusters hugged the surface closely enough to read as embedded in it.
    const dustGap = Math.max(1.4, R * 0.35);
    const dustStagger = 0.4 + R * 0.12;
    const phase = trng() * Math.PI * 2;
    dust.push({
      cluster, points, clickSphere, task,
      orbitR: R + (dustGap + (i % 5) * dustStagger) * DUST_QUAD_GAP[quad],
      speed: 0.12 + trng() * 0.12,
      quadSpeed: dustQuadSpeed(quad, overdue),
      phase,
      ang: phase, // accumulated orbit angle — advanced per-frame so speed can change live
      incline: (trng() - 0.5) * 0.9,
      anim: null, // {t} while crystallizing
    });
  });

  rec.impostor.visible = false;
  rec.detailBuilt = true;
}

function disposeThreeObject(o) {
  o.traverse((n) => {
    if (n.geometry) n.geometry.dispose();
    if (n.material) {
      if (n.material.map) n.material.map.dispose();
      n.material.dispose();
    }
  });
}

// Re-derives each planet's Eisenhower aggregate (rec.q1Count) and syncs its
// name label's ▲-badge. Called from scene-sync on every projects change, and
// hourly — urgency is time-derived, so quadrants flip at day boundaries with
// no data change at all.
function refreshPlanetPriorities(world, projects) {
  for (const p of projects) {
    const rec = world.planets.get(p.id);
    if (!rec) continue;
    rec.q1Count = projectPriority(p).q1;
    // motion-only cue (not brightness — dimming risks recreating the exact
    // contrast problem already fixed once): a stale world visibly slows its
    // own revolution and spin, reads as neglected without touching color.
    rec.staleFactor = isProjectStale(p) ? 0.4 : 1;
    if (rec.label) {
      const wanted = rec.q1Count > 0 ? `▲${rec.q1Count} · ${p.name}` : p.name;
      if (rec.label.textContent !== wanted) rec.label.textContent = wanted;
      rec.label.classList.toggle("q1", rec.q1Count > 0);
    }
    // live dust re-triage on the focused planet: importance/due edits don't
    // change signature() (no rebuild, by design), and d.task is a snapshot
    // from build time — look the task up fresh and update the agitation
    // speed, severity color/size, and label glyph. Seamless under
    // accumulated angles; orbit distance stays as built so nothing jumps.
    if (rec.detailBuilt) {
      for (const d of rec.dust) {
        const fresh = p.tasks.find((t) => t.id === d.task.id);
        if (fresh) {
          d.task = fresh;
          const quad = taskQuadrant(fresh);
          const overdue = taskOverdue(fresh);
          d.quadSpeed = dustQuadSpeed(quad, overdue);
          d.points.material.color.copy(dustQuadColor(quad, rec.pal, overdue));
          d.points.material.size = dustQuadSize(quad, overdue);
        }
      }
      if (rec.taskLabels) {
        for (const tl of rec.taskLabels) {
          if (tl.kind !== "dust") continue;
          const fresh = p.tasks.find((t) => t.id === tl.taskId);
          if (!fresh) continue;
          const quad = taskQuadrant(fresh);
          const overdue = taskOverdue(fresh);
          const wanted = dustQuadGlyph(quad, overdue) + fresh.title;
          if (tl.el.textContent !== wanted) tl.el.textContent = wanted;
          tl.el.classList.toggle("q1", quad === "q1");
          tl.el.classList.toggle("q3", quad === "q3");
          tl.el.classList.toggle("overdue", overdue);
        }
      }
    }
  }
}

function detachPlanetDetail(rec) {
  if (!rec.detailBuilt) return;
  rec.chunks.forEach((c) => { rec.spin.remove(c.mesh); disposeThreeObject(c.mesh); });
  rec.dust.forEach((d) => { rec.group.remove(d.cluster); disposeThreeObject(d.cluster); });
  rec.chunks = [];
  rec.dust = [];
  rec.impostor.visible = true;
  rec.detailBuilt = false;
}

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
  const galaxyLabelLayerRef = useRef(null); // plain DOM labels for planet names, updated imperatively per-frame
  const taskLabelLayerRef = useRef(null); // plain DOM labels for task names on the focused planet
  const [projects, setProjects] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [view, setView] = useState({ mode: "galaxy", projectId: null });
  const [selected, setSelected] = useState(null); // {projectId, taskId}
  const [hoverTip, setHoverTip] = useState(null);
  const [intakeOpen, setIntakeOpen] = useState(false);
  const [intakeMode, setIntakeMode] = useState("project");
  const [indexOpen, setIndexOpen] = useState(false);
  const [indexQuery, setIndexQuery] = useState("");
  const [debugOpen, setDebugOpen] = useState(false);
  const [dbgCount, setDbgCount] = useState(10);
  const [dbgConfirmClear, setDbgConfirmClear] = useState(false);
  const [, setDebugTick] = useState(0); // forces the debug panel to re-read live world/project data
  const [soundMode, setSoundMode] = useState("off"); // "off" | "ambient" | "classical"
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [confirmDeleteTask, setConfirmDeleteTask] = useState(null);
  const [saveNote, setSaveNote] = useState("");
  const [user, setUser] = useState(null); // Supabase auth user, or null when signed out / not configured
  const viewRef = useRef(view);
  viewRef.current = view;
  const intakeOpenRef = useRef(intakeOpen);
  intakeOpenRef.current = intakeOpen;
  const soundRef = useRef({ ready: false, synth: null, chime: null, timer: null });
  const dragRef = useRef({ down: false, moved: false, startX: 0, startY: 0, startTheta: 0, startAlpha: 0 });
  const reducedMotion = useRef(
    typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );
  // track live OS-setting changes — the ref is read per-frame in the animate
  // loop, so no re-render is needed, just keep it current
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const onChange = (e) => { reducedMotion.current = e.matches; };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    setConfirmDeleteTask(null);
  }, [selected]);

  /* ---------- debug panel refresh ---------- */
  useEffect(() => {
    if (!debugOpen) return;
    const id = setInterval(() => setDebugTick((t) => t + 1), 300);
    return () => clearInterval(id);
  }, [debugOpen]);

  // form state
  const [fProjName, setFProjName] = useState("");
  const [fProjDesc, setFProjDesc] = useState("");
  const [fProjArch, setFProjArch] = useState("general");
  const [fTaskProject, setFTaskProject] = useState("");
  const [fTaskTitle, setFTaskTitle] = useState("");
  const [fTaskNotes, setFTaskNotes] = useState("");
  const [fTaskDue, setFTaskDue] = useState("");
  const [fImportant, setFImportant] = useState(true);

  // archetype-driven intake prefills: picking a world resets the Important
  // checkbox to that world's default, and exec (fire-drill) worlds pre-fill
  // due = today so urgent capture stays title + Enter. Runs on world change
  // only — it never overwrites edits made after the world is chosen. The
  // just-created-world path can't rely on this effect (projectsRef doesn't
  // include the new project yet when it fires), so addProject applies the
  // same prefills itself.
  useEffect(() => {
    const proj = projectsRef.current.find((p) => p.id === fTaskProject);
    if (!proj) return;
    const arch = archetypeFor(proj);
    setFImportant(arch.importantDefault);
    if (arch.dueToday) setFTaskDue(todayISO());
  }, [fTaskProject]);

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
        if (user) {
          saveCloudProjects(user.id, projects)
            .then(() => { setSaveNote("synced"); setTimeout(() => setSaveNote(""), 1500); })
            .catch(() => { setSaveNote("sync failed"); setTimeout(() => setSaveNote(""), 2500); });
        } else {
          setSaveNote("saved");
          setTimeout(() => setSaveNote(""), 1500);
        }
      } catch (e) {
        setSaveNote("save failed");
      }
    }, 600);
    return () => clearTimeout(t);
  }, [projects, loaded, user]);

  /* ---------- cloud sync (Supabase, optional) ---------- */
  // Hydrates auth state; entirely inert if VITE_SUPABASE_* env vars are
  // absent (supabase client is null), so this is a no-op for anyone running
  // the app without cloud sync configured.
  useEffect(() => {
    if (!supabase) return;
    // If we just landed from an OAuth redirect, supabase-js's own hash
    // processing can occasionally still be settling when this first
    // getSession() resolves, showing signed-out until something (a manual
    // refresh) re-triggers it. A short delayed re-check closes that gap.
    const hadAuthRedirect = window.location.hash.includes("access_token");
    const clearHash = () => {
      if (window.location.hash) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
      }
    };
    supabase.auth.getSession().then(({ data }) => setUser(data.session?.user ?? null));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      clearHash(); // whatever supabase-js did or didn't clean up, don't leave token debris in the URL
    });
    let retry;
    if (hadAuthRedirect) {
      retry = setTimeout(() => {
        supabase.auth.getSession().then(({ data }) => setUser(data.session?.user ?? null));
      }, 400);
    }
    return () => { sub.subscription.unsubscribe(); clearTimeout(retry); };
  }, []);

  // The backend is authoritative once signed in — a user can never be "out
  // of sync" because there's only one copy that matters. Runs once per
  // sign-in (keyed on user+loaded, not on projects — must NOT re-fire on
  // every subsequent edit). No cloud row yet (brand-new account) → today's
  // projects become the seed, the one case with nothing to defer to yet.
  // Otherwise the cloud copy replaces whatever's showing, unconditionally —
  // no comparison against local, no prompt.
  useEffect(() => {
    if (!user || !loaded || !supabase) return;
    let cancelled = false;
    (async () => {
      try {
        const cloudProjects = await loadCloudProjects(user.id);
        if (cancelled) return;
        if (cloudProjects === null) {
          await saveCloudProjects(user.id, projectsRef.current);
        } else {
          setProjects(cloudProjects);
          // mirror immediately so a previous account's cache can't linger
          try { localStorage.setItem("orrery_galaxy_v1", JSON.stringify({ projects: cloudProjects })); } catch (e) { /* best-effort cache */ }
        }
      } catch (e) {
        setSaveNote("sync failed");
        setTimeout(() => setSaveNote(""), 2500);
      }
    })();
    return () => { cancelled = true; };
  }, [user, loaded]);

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
    // Far plane raised from 600: at 100+ planets the packed layout extent plus
    // a fully zoomed-out free-orbit camera can both sit well past the old
    // limit, which would clip distant planets out of existence entirely.
    const camera = new THREE.PerspectiveCamera(52, mount.clientWidth / mount.clientHeight, 0.1, 3000);
    camera.position.set(0, 34, 96);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x04060f, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x30395c, 1.8));
    const rim = new THREE.PointLight(0x6a7dff, 0.9, 200);
    rim.position.set(-40, -10, -30);
    scene.add(rim);

    // the star every project-planet orbits — self-lit (MeshBasicMaterial ignores
    // scene lights, appropriate for something that IS the light source), warm
    // and bright on purpose as the one non-cool object in the palette, and the
    // actual light planets are lit by (a point light replaces the old fixed
    // "sun" directional light now that there's a real object to hang it off of).
    const sunGeo = new THREE.IcosahedronGeometry(STAR_RADIUS, 24);
    const sunMat = new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(0.11, 0.85, 0.74) });
    const sunMesh = new THREE.Mesh(sunGeo, sunMat);
    scene.add(sunMesh);
    const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture("255,238,210"),
      transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    sunGlow.scale.set(STAR_RADIUS * 6, STAR_RADIUS * 6, 1);
    scene.add(sunGlow);
    const sunLight = new THREE.PointLight(0xfff2df, 3.2, 0, 1.15);
    sunLight.position.set(0, 0, 0);
    scene.add(sunLight);

    // starfield — pushed well beyond any plausible planet orbit or the sun's
    // own glow (400-580, vs. planets/sun living well under 300 in practice) so
    // background stars never read as floating in front of foreground objects.
    // They're meant to be the impossibly-distant backdrop, not nearby bodies.
    const starGeo = new THREE.BufferGeometry();
    const starN = 2200;
    const sp = new Float32Array(starN * 3);
    const sc = new Float32Array(starN * 3);
    for (let i = 0; i < starN; i++) {
      const r = 400 + Math.random() * 180;
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
      zoom: { planet: 1, galaxy: 1 }, // user-controlled distance multiplier, kept per mode
      galaxyBase: { radius: 46 }, // unzoomed galaxy-view orbit radius; zoom/orbit scale off this
      // free-orbit camera around the star in galaxy view, same spherical-coordinate
      // model as planet-mode `orbit` above, just targeting the origin instead of
      // a planet. Initial alpha matches the old fixed establishing-shot angle.
      galaxyOrbit: { theta: 0, alpha: Math.atan2(14, 44), dragging: false },
    };
    worldRef.current = world;

    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // Scroll-to-zoom (and trackpad pinch, which browsers report as wheel
    // events with ctrlKey set and larger deltaY). Native listener, not React's
    // onWheel — React 17+ attaches wheel as a passive listener at the root,
    // so preventDefault() inside a JSX onWheel handler silently does nothing.
    const onWheel = (e) => {
      e.preventDefault();
      const mode = viewRef.current.mode;
      if (mode === "planet") {
        const sens = e.ctrlKey ? ZOOM_PINCH_SENSITIVITY : ZOOM_WHEEL_SENSITIVITY;
        world.zoom.planet = clamp(world.zoom.planet * (1 + e.deltaY * sens), PLANET_ZOOM_MIN, PLANET_ZOOM_MAX);
      } else if (mode === "galaxy") {
        const sens = e.ctrlKey ? ZOOM_PINCH_SENSITIVITY : ZOOM_WHEEL_SENSITIVITY;
        world.zoom.galaxy = clamp(world.zoom.galaxy * (1 + e.deltaY * sens), GALAXY_ZOOM_MIN, GALAXY_ZOOM_MAX);
        // camTarget itself is recomputed every frame in the animate loop from
        // galaxyOrbit + this zoom factor — nothing more to do here.
      }
    };
    mount.addEventListener("wheel", onWheel, { passive: false });

    const labelVec = new THREE.Vector3(); // scratch, reused every frame for projection
    // How many planets before galaxy labels start thinning by camera distance
    // instead of all showing at once — small galaxies never hit this.
    const LABEL_ALWAYS_COUNT = 18;

    let raf;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      const rawDt = world.clock.getDelta();
      const dt = Math.min(rawDt, 0.05);
      const t = world.clock.elapsedTime;
      const motion = reducedMotion.current ? 0.25 : 1;
      if (rawDt > 0) world.fps = world.fps ? world.fps * 0.9 + (1 / rawDt) * 0.1 : 1 / rawDt;

      stars.rotation.y += dt * 0.004 * motion;
      const sunPulse = 1 + Math.sin(t * 0.9) * 0.03 * motion;
      sunMesh.scale.setScalar(sunPulse);
      sunGlow.material.opacity = 0.85 + Math.sin(t * 0.9) * 0.05 * motion;

      const v = viewRef.current;
      for (const [pid, rec] of world.planets) {
        // stale worlds visibly slow their own revolution and spin — a
        // motion-only "neglected" cue, not a brightness one (see staleFactor)
        const staleFactor = rec.staleFactor || 1;
        rec.orbitAngle += dt * rec.orbitSpeed * motion * staleFactor;
        rec.group.position.set(
          Math.cos(rec.orbitAngle) * rec.orbitRadius,
          rec.orbitY,
          Math.sin(rec.orbitAngle) * rec.orbitRadius
        );
        const focused = v.mode === "planet" && v.projectId === pid;
        const target = focused ? 1 : 0;
        // reduced motion means SNAP to the final state, not slow-mo — the
        // scatter/reform tween is one of the most vestibular-heavy effects
        if (reducedMotion.current) rec.explode = target;
        else rec.explode += (target - rec.explode) * Math.min(1, dt * 2.4);
        rec.spinAngle += dt * rec.spinSpeed * motion * staleFactor * (1 - 0.75 * rec.explode);
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
        rec.dust.forEach((d) => {
          // orbit angle is ACCUMULATED (not derived from absolute time) so
          // quadSpeed can change on live re-triage without the cluster
          // teleporting to a new time-derived position.
          if (d.anim) {
            d.anim.t = Math.min(1, d.anim.t + dt / 1.35);
            const k = 1 - d.anim.t;
            const ang = d.ang + d.anim.t * 4.0; // base angle frozen at collapse start, spiral term unchanged
            const rr = d.orbitR * k * k;
            d.cluster.position.set(
              Math.cos(ang) * rr,
              Math.sin(ang * 0.7 + d.incline) * rr * 0.35,
              Math.sin(ang) * rr
            );
            d.points.material.opacity = 0.95 * k + 0.05;
            d.cluster.scale.setScalar(Math.max(0.05, k));
          } else {
            d.ang += dt * d.speed * (d.quadSpeed || 1) * motion;
            d.cluster.position.set(
              Math.cos(d.ang) * d.orbitR,
              Math.sin(d.ang * 0.7 + d.incline) * d.orbitR * 0.35,
              Math.sin(d.ang) * d.orbitR
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
          const radius = Math.hypot(baseHeight, baseDist) * world.zoom.planet;
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

      // free-orbit around the star (user-controlled via drag), same spherical
      // model as the planet-mode block above, just targeting the origin at a
      // zoom-scaled radius instead of a planet's position.
      if (v.mode === "galaxy" && world.galaxyOrbit) {
        const radius = world.galaxyBase.radius * world.zoom.galaxy;
        const alpha = world.galaxyOrbit.alpha;
        const theta = world.galaxyOrbit.theta;
        const horizontalDist = radius * Math.cos(alpha);
        world.camTarget.set(
          horizontalDist * Math.sin(theta),
          radius * Math.sin(alpha),
          horizontalDist * Math.cos(theta)
        );
        world.lookTarget.set(0, 0, 0);
      }

      // camera glide — instant follow while actively dragging (either mode),
      // eased lerp otherwise so clicks/zoom/focus changes glide into place.
      if ((v.mode === "planet" && world.orbit?.dragging) || (v.mode === "galaxy" && world.galaxyOrbit?.dragging)) {
        world.camPos.set(world.camTarget.x, world.camTarget.y, world.camTarget.z);
      } else {
        world.camPos.lerp(world.camTarget, Math.min(1, dt * 2.2));
      }
      camera.position.copy(world.camPos);
      world.look.lerp(world.lookTarget, Math.min(1, dt * 2.6));
      camera.lookAt(world.look);

      renderer.render(scene, camera);

      // planet-name labels — plain DOM nodes projected to screen space each
      // frame. Past LABEL_ALWAYS_COUNT planets, thin by distance from camera
      // (the near hemisphere lights up, far side fades) rather than showing
      // every name at once, which would just read as an overlapping wall of
      // text at 100+ planets.
      const galaxyLayer = galaxyLabelLayerRef.current;
      if (galaxyLayer) {
        const showGalaxyLabels = v.mode === "galaxy";
        const thinning = world.planets.size > LABEL_ALWAYS_COUNT;
        const camRadius = world.galaxyBase.radius * world.zoom.galaxy;
        for (const [, rec] of world.planets) {
          const el = rec.label;
          if (!el) continue;
          if (!showGalaxyLabels) { el.style.display = "none"; continue; }
          labelVec.copy(rec.group.position).project(camera);
          if (labelVec.z > 1 || labelVec.z < -1) { el.style.display = "none"; continue; }
          let opacity = 1;
          // Q1 planets defeat thinning entirely: at any zoom, the only names
          // guaranteed readable are the ones that need attention ("fly to me
          // first"). The ▲-count in the label itself carries the signal by
          // shape, not color alone.
          if (thinning && !(rec.q1Count > 0)) {
            const dist = camera.position.distanceTo(rec.group.position);
            const near = camRadius * 0.55, far = camRadius * 1.35;
            opacity = clamp(1 - (dist - near) / (far - near), 0, 1);
          }
          if (opacity <= 0.03) { el.style.display = "none"; continue; }
          const x = (labelVec.x * 0.5 + 0.5) * mount.clientWidth;
          const y = (-labelVec.y * 0.5 + 0.5) * mount.clientHeight;
          el.style.display = "block";
          el.style.opacity = opacity.toFixed(2);
          el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        }
      }

      // task-name labels — same projection, scoped to the focused planet's
      // own chunks/dust, tracking their live (possibly still-exploding or
      // still-orbiting) world position rather than a fixed offset.
      const taskLayer = taskLabelLayerRef.current;
      if (taskLayer) {
        const rec = v.mode === "planet" ? world.planets.get(v.projectId) : null;
        if (rec?.taskLabels) {
          const explodeAmt = rec.explode * (rec.R * 0.55 + 0.45); // matches the chunk-scatter math above
          for (const tl of rec.taskLabels) {
            if (tl.kind === "chunk") {
              labelVec.copy(tl.dir).multiplyScalar(rec.R + explodeAmt);
              rec.spin.localToWorld(labelVec);
            } else {
              tl.cluster.getWorldPosition(labelVec);
            }
            labelVec.project(camera);
            if (labelVec.z > 1 || labelVec.z < -1) { tl.el.style.display = "none"; continue; }
            const x = (labelVec.x * 0.5 + 0.5) * mount.clientWidth;
            const y = (-labelVec.y * 0.5 + 0.5) * mount.clientHeight;
            tl.el.style.display = "block";
            tl.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
          }
        }
      }
    };
    animate();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      mount.removeEventListener("wheel", onWheel);
      for (const [, rec] of world.planets) disposeRecord(rec);
      starGeo.dispose();
      sunGeo.dispose(); sunMat.dispose();
      sunGlow.material.map.dispose(); sunGlow.material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      worldRef.current = null;
    };
  }, []);

  // task-name labels for a focused planet's chunks/dust — plain DOM nodes,
  // rebuilt any time that planet's detail mesh is (re)built: on first focus,
  // and again whenever a task completes/changes while it's still focused.
  const clearTaskLabels = (rec) => {
    if (!rec.taskLabels) return;
    rec.taskLabels.forEach((tl) => tl.el.remove());
    rec.taskLabels = null;
  };
  const buildTaskLabels = (rec) => {
    const layer = taskLabelLayerRef.current;
    if (!layer || rec.taskLabels) return;
    rec.taskLabels = [];
    rec.chunks.forEach((c) => {
      if (!c.task) return;
      const el = document.createElement("div");
      el.className = "task-label";
      el.textContent = c.task.title;
      layer.appendChild(el);
      // chunk.mesh's own origin sits at the planet's center (its geometry's
      // vertices carry the actual surface offsets, not the mesh transform) —
      // dir is the chunk's surface direction, needed to anchor the label at
      // the visible fragment rather than the planet's core.
      rec.taskLabels.push({ el, kind: "chunk", dir: c.dir });
    });
    rec.dust.forEach((d) => {
      const el = document.createElement("div");
      const quad = taskQuadrant(d.task);
      const overdue = taskOverdue(d.task);
      el.className = "task-label" + (quad === "q1" ? " q1" : quad === "q3" ? " q3" : "") + (overdue ? " overdue" : "");
      el.textContent = dustQuadGlyph(quad, overdue) + d.task.title;
      layer.appendChild(el);
      rec.taskLabels.push({ el, kind: "dust", cluster: d.cluster, taskId: d.task.id });
    });
  };

  /* ---------- scene sync ---------- */
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    const keep = new Set(projects.map((p) => p.id));
    for (const [id, rec] of [...world.planets]) {
      if (!keep.has(id)) {
        world.scene.remove(rec.group);
        disposeRecord(rec);
        rec.label?.remove();
        clearTaskLabels(rec);
        world.planets.delete(id);
      }
    }
    const orbits = computeLayout(projects);
    let outer = STAR_RADIUS * 4; // floor so the star itself always frames cleanly, even with 0-1 projects
    projects.forEach((p, i) => {
      const orbit = orbits[i];
      outer = Math.max(outer, Math.hypot(orbit.radius, orbit.y) + radiusFor(p));
      const sig = signature(p);
      const isFocused = viewRef.current.mode === "planet" && viewRef.current.projectId === p.id;
      let rec = world.planets.get(p.id);
      if (rec && rec.sig !== sig) {
        const explode = rec.explode, spinAngle = rec.spinAngle, orbitAngle = rec.orbitAngle;
        world.scene.remove(rec.group);
        disposeRecord(rec);
        rec.label?.remove();
        clearTaskLabels(rec);
        rec = null;
        const fresh = buildPlanetShell(p, orbit);
        fresh.explode = explode;
        fresh.spinAngle = spinAngle;
        fresh.orbitAngle = orbitAngle; // keep revolving smoothly through a rebuild, don't snap back to phase
        world.scene.add(fresh.group);
        world.planets.set(p.id, fresh);
        if (galaxyLabelLayerRef.current) {
          fresh.label = document.createElement("div");
          fresh.label.className = "planet-label";
          fresh.label.textContent = p.name;
          galaxyLabelLayerRef.current.appendChild(fresh.label);
        }
        if (isFocused) { attachPlanetDetail(fresh, p); buildTaskLabels(fresh); } // e.g. completing a task on the focused planet
      } else if (!rec) {
        const fresh = buildPlanetShell(p, orbit);
        world.scene.add(fresh.group);
        world.planets.set(p.id, fresh);
        if (galaxyLabelLayerRef.current) {
          fresh.label = document.createElement("div");
          fresh.label.className = "planet-label";
          fresh.label.textContent = p.name;
          galaxyLabelLayerRef.current.appendChild(fresh.label);
        }
        if (isFocused) attachPlanetDetail(fresh, p);
      } else {
        // signature unchanged, but the layout may have shifted (e.g. an earlier
        // planet resized) — adopt the new radius/y, keep the current angle so
        // motion stays continuous instead of jumping.
        rec.orbitRadius = orbit.radius;
        rec.orbitY = orbit.y;
        rec.orbitSpeed = orbitSpeedFor(orbit.radius);
      }
    });
    world.galaxyExtent = outer;
    // owns label text (name + ▲-badge) for every planet, including renames —
    // importance/due edits don't touch signature(), so this unconditional
    // pass is what keeps triage state fresh without planet rebuilds.
    refreshPlanetPriorities(world, projects);
  }, [projects]);

  /* ---------- hourly triage refresh (urgency is time-derived) ---------- */
  useEffect(() => {
    const id = setInterval(() => {
      const world = worldRef.current;
      if (world) refreshPlanetPriorities(world, projectsRef.current);
    }, 60 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  /* ---------- camera base radius (galaxy) ---------- */
  // Only the base orbit RADIUS is computed here — the actual camTarget is
  // derived every frame in the animate loop from this radius combined with
  // the user's free-orbit theta/alpha and zoom, the same split planet-mode
  // already uses (baseDist here, orbit angles in the render loop).
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    if (view.mode === "galaxy") {
      // Distance driven by the actual packed layout extent (which now grows
      // sub-linearly with project count post-repacking, see computeLayout),
      // not a separate linear-in-n term — capped so a big galaxy (dozens of
      // projects, stress-test batches) doesn't push the camera back
      // indefinitely once the establishing shot stops gaining from it.
      const far = Math.max(30, (world.galaxyExtent || 20) * 1.6);
      const radius = Math.min(320, far);
      world.galaxyBase.radius = radius;
    }
  }, [view.mode, projects]);

  /* ---------- orbit init on planet focus (drag-to-rotate state) ---------- */
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    // Only the focused project needs its expensive detail mesh built — every
    // other planet reverts to its cheap galaxy-view impostor. detachPlanetDetail
    // is a no-op for records that already have no detail attached.
    for (const [id, rec] of world.planets) {
      if (!(view.mode === "planet" && id === view.projectId)) {
        detachPlanetDetail(rec);
        clearTaskLabels(rec);
      }
    }
    if (view.mode === "planet") {
      const rec = world.planets.get(view.projectId);
      const R = rec ? rec.R : 1.5;
      const baseHeight = R * 0.9;
      const baseDist = R * 3.4 + 4.2;
      world.orbit = { theta: 0, alpha: Math.atan2(baseHeight, baseDist), dragging: false };
      if (rec) {
        const project = projectsRef.current.find((p) => p.id === view.projectId);
        if (project) attachPlanetDetail(rec, project);
        buildTaskLabels(rec);
      }
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

  /* ---------- arrow-key planet cycling / escape-to-galaxy ---------- */
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (intakeOpenRef.current) return;

      if (e.key === "Escape") {
        if (viewRef.current.mode === "planet") {
          e.preventDefault();
          setView({ mode: "galaxy", projectId: null });
          setSelected(null);
          setHoverTip(null);
        }
        return;
      }

      // triage key — fly to the world that most needs attention; press again
      // to cycle through every ▲ world (same order as the Index sort).
      if (e.key === "1") {
        const q1List = projectsRef.current
          .map((p) => ({ p, q1: projectPriority(p).q1 }))
          .filter((x) => x.q1 > 0)
          .sort((a, b) => b.q1 - a.q1)
          .map((x) => x.p);
        if (!q1List.length) return;
        e.preventDefault();
        const v = viewRef.current;
        const idx = v.mode === "planet" ? q1List.findIndex((p) => p.id === v.projectId) : -1;
        const next = q1List[(idx + 1) % q1List.length];
        setView({ mode: "planet", projectId: next.id });
        setSelected(null);
        setHoverTip(null);
        return;
      }

      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const ps = projectsRef.current;
      if (!ps.length) return;
      e.preventDefault();
      const v = viewRef.current;
      const idx = v.mode === "planet" ? ps.findIndex((p) => p.id === v.projectId) : -1;
      const nextIdx = e.key === "ArrowRight"
        ? (idx === -1 ? 0 : (idx + 1) % ps.length)
        : (idx === -1 ? ps.length - 1 : (idx - 1 + ps.length) % ps.length);
      setView({ mode: "planet", projectId: ps[nextIdx].id });
      setSelected(null);
      setHoverTip(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const onCanvasPointerDown = useCallback((e) => {
    const world = worldRef.current;
    const drag = dragRef.current;
    drag.down = true;
    drag.moved = false;
    drag.startX = e.clientX;
    drag.startY = e.clientY;
    if (viewRef.current.mode === "planet" && world?.orbit) {
      drag.startTheta = world.orbit.theta;
      drag.startAlpha = world.orbit.alpha;
    } else if (viewRef.current.mode === "galaxy" && world?.galaxyOrbit) {
      drag.startTheta = world.galaxyOrbit.theta;
      drag.startAlpha = world.galaxyOrbit.alpha;
    }
    try { mountRef.current?.setPointerCapture?.(e.pointerId); } catch (err) { /* best-effort */ }
  }, []);

  const onCanvasPointerUp = useCallback((e) => {
    const world = worldRef.current;
    try { mountRef.current?.releasePointerCapture?.(e.pointerId); } catch (err) { /* best-effort */ }
    const drag = dragRef.current;
    drag.down = false;
    if (world?.orbit) world.orbit.dragging = false;
    if (world?.galaxyOrbit) world.galaxyOrbit.dragging = false;
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
      } else if (viewRef.current.mode === "galaxy" && world.galaxyOrbit) {
        world.galaxyOrbit.dragging = true;
        world.galaxyOrbit.theta = drag.startTheta - dx * ORBIT_SENSITIVITY;
        world.galaxyOrbit.alpha = clamp(drag.startAlpha + dy * ORBIT_SENSITIVITY, -ORBIT_ALPHA_LIMIT, ORBIT_ALPHA_LIMIT);
      }
      mount.style.cursor = "grabbing";
      setHoverTip(null);
      return;
    }

    const hit = pick(e.clientX, e.clientY);
    mount.style.cursor = hit ? "pointer" : "grab";
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
    const p = { id: uid(), name, desc: fProjDesc.trim(), archetype: fProjArch, createdAt: Date.now(), tasks: [] };
    setProjects((ps) => [...ps, p]);
    setFProjName(""); setFProjDesc(""); setFProjArch("general");
    setIntakeMode("task");
    setFTaskProject(p.id);
    // apply this archetype's task prefills directly — the world-change effect
    // can't see the new project in projectsRef yet (see its comment)
    const arch = archetypeFor(p);
    setFImportant(arch.importantDefault);
    setFTaskDue(arch.dueToday ? todayISO() : "");
  }, [fProjName, fProjDesc, fProjArch]);

  const addTask = useCallback(() => {
    const title = fTaskTitle.trim();
    if (!title || !fTaskProject) return;
    setProjects((ps) => ps.map((p) => p.id === fTaskProject
      ? { ...p, tasks: [...p.tasks, { id: uid(), title, notes: fTaskNotes.trim(), due: fTaskDue || null, important: fImportant, done: false, createdAt: Date.now(), completedAt: null }] }
      : p));
    // reset back to the selected world's archetype defaults (not bare blanks),
    // so back-to-back captures on a fire-drill world keep their prefills
    const arch = archetypeFor(projectsRef.current.find((p) => p.id === fTaskProject));
    setFTaskTitle(""); setFTaskNotes("");
    setFTaskDue(arch.dueToday ? todayISO() : "");
    setFImportant(arch.importantDefault);
  }, [fTaskTitle, fTaskNotes, fTaskDue, fTaskProject, fImportant]);

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
        // bloom flash scales a sprite up to ~14x — skip under reduced motion
        // (same pattern as the dust-collapse skip below); the chime remains
        // as the completion acknowledgment
        if (!reducedMotion.current) spawnBloom(projectId, projectDone);
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

  // re-triage from the detail panel — signature() ignores `important`, so this
  // re-derives labels/index without triggering an expensive planet rebuild.
  const setTaskImportant = useCallback((projectId, taskId, important) => {
    setProjects((ps) => ps.map((p) => p.id === projectId
      ? { ...p, tasks: p.tasks.map((t) => t.id === taskId ? { ...t, important } : t) }
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

  /* ---------- debug/simulation actions (bulk data ops for fast manual QA) ---------- */
  // open debug tasks get a deterministic quadrant mix (¼ overdue, ¼ due in
  // 2 days, rest undated; every 3rd not-important) so stress-test galaxies
  // exercise all four Eisenhower quadrants instead of reading uniformly Q2.
  const dbgMakeTasks = (count, done) => Array.from({ length: count }, (_, i) => ({
    id: uid(), title: `Debug task ${i + 1}`, notes: "",
    due: done ? null : (i % 4 === 0 ? todayISO(-1) : i % 4 === 1 ? todayISO(2) : null),
    important: i % 3 !== 0,
    done, createdAt: Date.now(), completedAt: done ? Date.now() : null,
  }));
  const dbgAddOpenTasks = useCallback((projectId, count) => {
    setProjects((ps) => ps.map((p) => p.id !== projectId ? p : { ...p, tasks: [...p.tasks, ...dbgMakeTasks(count, false)] }));
  }, []);
  const dbgCompleteAllOpen = useCallback((projectId) => {
    setProjects((ps) => ps.map((p) => p.id !== projectId ? p : {
      ...p, tasks: p.tasks.map((t) => t.done ? t : { ...t, done: true, completedAt: Date.now() }),
    }));
  }, []);
  const dbgUncompleteAll = useCallback((projectId) => {
    setProjects((ps) => ps.map((p) => p.id !== projectId ? p : {
      ...p, tasks: p.tasks.map((t) => ({ ...t, done: false, completedAt: null })),
    }));
  }, []);
  const dbgRandomProjectNames = ["Nebula Draft", "Comet Log", "Asteroid Notes", "Void Sketch", "Orbit Plan", "Quasar Memo", "Ion Trail", "Dust Ledger"];
  const dbgMakeRandomProject = (label) => {
    const done = Math.floor(Math.random() * 45);
    const open = Math.floor(Math.random() * 6);
    const name = label || `${dbgRandomProjectNames[Math.floor(Math.random() * dbgRandomProjectNames.length)]} ${Math.floor(Math.random() * 1000)}`;
    return { id: uid(), name, desc: "debug-generated", createdAt: Date.now(), tasks: [...dbgMakeTasks(done, true), ...dbgMakeTasks(open, false)] };
  };
  const dbgAddRandomProject = useCallback(() => {
    setProjects((ps) => [...ps, dbgMakeRandomProject()]);
  }, []);
  const dbgStressTest = useCallback(() => {
    setProjects((ps) => [...ps, ...Array.from({ length: 8 }, (_, i) => dbgMakeRandomProject(`Stress ${i}`))]);
  }, []);
  const dbgClearAll = useCallback(() => {
    setDbgConfirmClear(false);
    setSelected(null);
    setView({ mode: "galaxy", projectId: null });
    setProjects([]);
  }, []);

  /* ---------- derived ---------- */
  const focusProject = view.mode === "planet" ? projects.find((p) => p.id === view.projectId) : null;
  const selTask = selected
    ? projects.find((p) => p.id === selected.projectId)?.tasks.find((t) => t.id === selected.taskId)
    : null;
  const overdue = taskOverdue;

  /* ---------- debug panel data (re-read live each debugTick while open) ---------- */
  const dbgWorld = worldRef.current;
  const dbgRec = dbgWorld && view.mode === "planet" ? dbgWorld.planets.get(view.projectId) : null;
  const dbgTotalDone = projects.reduce((s, p) => s + p.tasks.filter((t) => t.done).length, 0);
  const dbgTotalTasks = projects.reduce((s, p) => s + p.tasks.length, 0);
  const v3 = (v) => v ? `${v.x.toFixed(1)}, ${v.y.toFixed(1)}, ${v.z.toFixed(1)}` : "—";

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
      <div ref={galaxyLabelLayerRef} className="label-layer" />
      <div ref={taskLabelLayerRef} className="label-layer" />
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
            <div className="build-tag">build {BUILD_SHA} · {BUILD_TIME_LABEL}</div>
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
              ? "A newborn core. Add tasks to give it matter. · ← → to switch worlds"
              : "Tap a fragment to recall its work · tap orbiting stardust to view open tasks · drag to rotate · scroll to zoom · ← → to switch worlds · 1 → next ▲"}
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
            {!selTask.done && (
              <div className={"d-row d-quadrant" + (taskQuadrant(selTask) === "q1" ? " warn" : "")}>
                {overdue(selTask)
                  ? {
                      q1: "▲ Do first — important & OVERDUE",
                      q3: "△ Contain — OVERDUE, not important",
                    }[taskQuadrant(selTask)]
                  : {
                      q1: "▲ Do first — important & due soon",
                      q2: "Scheduled — important, not yet urgent",
                      q3: "△ Contain — due soon, not important",
                      q4: "Drifting — not important, no near date",
                    }[taskQuadrant(selTask)]}
                <button
                  className="ghost-btn sm"
                  onClick={() => setTaskImportant(selected.projectId, selTask.id, !taskImportant(selTask))}
                >
                  {taskImportant(selTask) ? "Demote: not important" : "Mark important"}
                </button>
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
        <button className="ghost-btn sm" onClick={() => setIndexOpen((o) => { if (o) setIndexQuery(""); return !o; })}>
          {indexOpen ? "Close index" : `Index · ${projects.length}`}
        </button>
        {indexOpen && (
          <div className="index">
            <input
              className="idx-search"
              autoFocus
              value={indexQuery}
              onChange={(e) => setIndexQuery(e.target.value)}
              placeholder="Search tasks or worlds…"
            />
            {indexQuery.trim() ? (() => {
              const { results, total } = searchTasks(projects, indexQuery);
              if (results.length === 0) {
                return <div className="idx-empty">No tasks match "{indexQuery.trim()}"</div>;
              }
              return (
                <>
                  {results.map(({ p, t }) => {
                    const quad = taskQuadrant(t);
                    const isOverdue = taskOverdue(t);
                    const sev = t.done ? "" : quad === "q1" ? " q1" : quad === "q3" ? " q3" : "";
                    return (
                      <button
                        key={t.id}
                        className="idx-task-row"
                        onClick={() => {
                          setView({ mode: "planet", projectId: p.id });
                          setSelected({ projectId: p.id, taskId: t.id });
                          setIndexOpen(false);
                          setIndexQuery("");
                        }}
                      >
                        <span className={"idx-task-title" + sev + (isOverdue ? " overdue" : "") + (t.done ? " done" : "")}>
                          {!t.done ? dustQuadGlyph(quad, isOverdue) : ""}{t.title}
                        </span>
                        <span className="idx-task-sub">{p.name}</span>
                      </button>
                    );
                  })}
                  {total > results.length && (
                    <div className="idx-legend">{total - results.length} more matches not shown</div>
                  )}
                </>
              );
            })() : (
              <>
                {projects.length === 0 && <div className="idx-empty">No worlds yet</div>}
                {projects
                  .map((p) => ({ p, pri: projectPriority(p) }))
                  .sort((a, b) => (b.pri.q1 - a.pri.q1) || ((b.pri.q3 > 0 ? 1 : 0) - (a.pri.q3 > 0 ? 1 : 0)))
                  .map(({ p, pri }) => {
                    const done = p.tasks.filter((t) => t.done).length;
                    const staleDays = projectStaleDays(p);
                    return (
                      <button key={p.id} className="idx-row" onClick={() => {
                        setView({ mode: "planet", projectId: p.id });
                        setSelected(null); setIndexOpen(false);
                      }}>
                        <span className="idx-name">{p.name}</span>
                        {pri.q1 > 0 && <span className="idx-badge q1">▲{pri.q1}</span>}
                        {pri.q1 === 0 && pri.q3 > 0 && <span className="idx-badge q3">△{pri.q3}</span>}
                        {staleDays >= STALE_DAYS && <span className="idx-badge idle">· {staleDays}d idle</span>}
                        <span className="idx-meta">{done}/{p.tasks.length}</span>
                      </button>
                    );
                  })}
                {projects.length > 0 && (
                  <div className="idx-legend">▲ important & due soon · △ due soon · press 1 to fly to ▲</div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* debug panel */}
      {debugOpen && (
        <div className="hud debug-panel">
          <div className="dbg-row"><span className="dbg-k">build</span> {BUILD_SHA} · {BUILD_TIME_LABEL}</div>
          <div className="dbg-row"><span className="dbg-k">view</span> {view.mode}{focusProject ? ` · ${focusProject.name}` : ""}</div>
          <div className="dbg-row"><span className="dbg-k">projects</span> {projects.length} · tasks {dbgTotalDone}/{dbgTotalTasks}</div>
          <div className="dbg-row"><span className="dbg-k">sound</span> {soundMode} · <span className="dbg-k">motion</span> {reducedMotion.current ? "reduced" : "full"}</div>
          <div className="dbg-row"><span className="dbg-k">fps</span> {dbgWorld?.fps ? dbgWorld.fps.toFixed(0) : "—"}</div>
          <div className="dbg-row"><span className="dbg-k">star</span> R={STAR_RADIUS} @ origin</div>
          <div className="dbg-row"><span className="dbg-k">camera</span> {v3(dbgWorld?.camPos)}</div>
          <div className="dbg-row"><span className="dbg-k">cam target</span> {v3(dbgWorld?.camTarget)}</div>
          <div className="dbg-row"><span className="dbg-k">zoom</span> planet {dbgWorld?.zoom.planet.toFixed(2)} · galaxy {dbgWorld?.zoom.galaxy.toFixed(2)}</div>
          {dbgRec && (
            <>
              <div className="dbg-row dbg-sep"><span className="dbg-k">focused planet</span></div>
              <div className="dbg-row"><span className="dbg-k">R</span> {dbgRec.R.toFixed(2)}</div>
              <div className="dbg-row"><span className="dbg-k">orbit r</span> {dbgRec.orbitRadius.toFixed(1)} · <span className="dbg-k">θ</span> {((dbgRec.orbitAngle * 180 / Math.PI) % 360).toFixed(1)}° · <span className="dbg-k">v</span> {dbgRec.orbitSpeed.toFixed(5)}</div>
              <div className="dbg-row"><span className="dbg-k">chunks</span> {dbgRec.chunks.length} · <span className="dbg-k">dust</span> {dbgRec.dust.length}</div>
              <div className="dbg-row"><span className="dbg-k">pos</span> {v3(dbgRec.group.position)}</div>
            </>
          )}

          <div className="dbg-row dbg-sep"><span className="dbg-k">simulate</span></div>
          <div className="dbg-row">
            <input
              type="number" className="dbg-input" min={1} max={300} value={dbgCount}
              onChange={(e) => setDbgCount(Math.max(1, Math.min(300, parseInt(e.target.value, 10) || 1)))}
            /> tasks per action
          </div>
          {focusProject ? (
            <div className="dbg-actions">
              <button className="ghost-btn sm" onClick={() => dbgAddOpenTasks(focusProject.id, dbgCount)}>+ add {dbgCount} open</button>
              <button className="ghost-btn sm" onClick={() => dbgCompleteAllOpen(focusProject.id)}>✦ complete all open</button>
              <button className="ghost-btn sm" onClick={() => dbgUncompleteAll(focusProject.id)}>↺ uncomplete all</button>
            </div>
          ) : (
            <div className="dbg-row" style={{ opacity: 0.6 }}>focus a planet for per-world actions</div>
          )}
          <div className="dbg-actions">
            <button className="ghost-btn sm" onClick={dbgAddRandomProject}>🎲 random world</button>
            <button className="ghost-btn sm" onClick={dbgStressTest}>🌌 stress test (+8 worlds)</button>
            {dbgConfirmClear ? (
              <span className="confirm">
                Clear ALL worlds?
                <button className="danger-btn sm" onClick={dbgClearAll}>Clear</button>
                <button className="ghost-btn sm" onClick={() => setDbgConfirmClear(false)}>Keep</button>
              </span>
            ) : (
              <button className="ghost-btn sm dim" onClick={() => setDbgConfirmClear(true)}>💣 clear all worlds</button>
            )}
          </div>
        </div>
      )}

      {/* bottom controls */}
      <div className="hud bottom-right">
        {saveNote && <span className="save-note">{saveNote}</span>}
        {supabase && (
          user ? (
            <button className="ghost-btn sm" onClick={() => supabase.auth.signOut()} title={user.email}>
              ☁ {user.email || "synced"}
            </button>
          ) : (
            <button
              className="ghost-btn sm"
              onClick={() => supabase.auth.signInWithOAuth({
                provider: "google",
                // Deliberately NOT window.location.href: if a retry happens while
                // the previous attempt's #access_token=... fragment is still in
                // the URL, Supabase appends a new fragment onto the old one
                // instead of replacing it, stacking into an ever-growing mess.
                // Always redirect to the clean base URL instead.
                options: { redirectTo: window.location.origin + window.location.pathname },
              })}
            >
              ☁ Sign in to sync
            </button>
          )
        )}
        <button className="ghost-btn sm" onClick={() => { setDebugOpen((o) => !o); setDbgConfirmClear(false); }} aria-label="Toggle debug panel">
          ⚙ debug
        </button>
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
                <label className="f-label" htmlFor="parch">Work type</label>
                <select id="parch" className="f-input" value={fProjArch} onChange={(e) => setFProjArch(e.target.value)}>
                  {ARCHETYPES.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
                </select>
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
                <label className="f-check">
                  <input type="checkbox" checked={fImportant} onChange={(e) => setFImportant(e.target.checked)} />
                  <span>Important — due dates on important tasks surface them as <b>▲ do first</b></span>
                </label>
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
.label-layer { position: absolute; inset: 0; z-index: 4; pointer-events: none; overflow: hidden; }
.planet-label {
  position: absolute; left: 0; top: 0; pointer-events: none;
  font-family: 'Cormorant Garamond', serif; font-size: 13px; letter-spacing: 0.02em;
  color: var(--ink-dim); text-shadow: 0 0 8px rgba(120,150,255,0.6);
  white-space: nowrap; will-change: transform, opacity;
}
/* Q1 planets — the ▲count prefix carries the signal by shape; the warm tint
   and brighter weight reinforce it but are never the sole channel. */
.planet-label.q1 { color: var(--warn); font-size: 13.5px; text-shadow: 0 0 10px rgba(232,197,138,0.55); }
.task-label {
  position: absolute; left: 0; top: 0; pointer-events: none;
  font-family: 'Cormorant Garamond', serif; font-size: 11px; opacity: 0.85;
  color: var(--ink-dim); text-shadow: 0 0 6px rgba(120,150,255,0.5);
  white-space: nowrap; will-change: transform;
}
/* severity tint mirrors the dust cluster's color; the ▲/△ glyph carries the
   same rank by shape for color-blind users */
.task-label.q1 { color: #ffab66; opacity: 1; font-size: 11.5px; text-shadow: 0 0 8px rgba(255,171,102,0.5); }
.task-label.q3 { color: #e8dc8a; text-shadow: 0 0 6px rgba(232,220,138,0.4); }
/* overdue escalates the same slot further — crimson/burnt-orange, a touch
   larger — an extra notch on top of the base q1/q3 tint above */
.task-label.q1.overdue { color: #ff5a3c; font-size: 12px; text-shadow: 0 0 9px rgba(255,90,60,0.6); }
.task-label.q3.overdue { color: #d98f3c; text-shadow: 0 0 7px rgba(217,143,60,0.45); }
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
.build-tag { font-family: monospace; font-size: 10px; color: var(--ink-dim); opacity: 0.55; margin-top: 6px; }

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
.d-quadrant { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 6px; }
.d-quadrant .ghost-btn.sm { font-size: 11px; padding: 4px 8px; }
.d-actions { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }

.empty {
  left: 50%; top: 50%; transform: translate(-50%, -50%);
  text-align: center; max-width: 92vw;
}
.empty-title { font-family: 'Cormorant Garamond', serif; font-size: 34px; font-weight: 500; }
.empty-sub { color: var(--ink-dim); margin: 8px 0 18px; font-size: 14px; }

.index-wrap { left: 18px; bottom: 18px; display: flex; flex-direction: column-reverse; align-items: flex-start; gap: 8px; }

.debug-panel {
  top: 18px; right: 18px; width: min(300px, 80vw); max-height: 70vh; overflow: auto;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 14px; backdrop-filter: blur(10px);
  font-family: monospace; font-size: 11.5px; line-height: 1.7; color: var(--ink-dim);
}
.dbg-row { white-space: nowrap; }
.dbg-k { color: var(--accent); }
.dbg-sep { margin-top: 8px; padding-top: 6px; border-top: 1px solid var(--line); }
.dbg-input {
  width: 60px; background: rgba(6,10,26,0.8); border: 1px solid var(--line);
  border-radius: 6px; color: var(--ink); font-family: inherit; font-size: 11.5px;
  padding: 2px 6px; margin-right: 4px;
}
.dbg-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.dbg-actions .ghost-btn.sm { font-size: 11px; padding: 5px 9px; white-space: nowrap; }
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
.idx-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.idx-meta { color: var(--ink-dim); font-size: 12px; flex-shrink: 0; }
.idx-empty { color: var(--ink-dim); font-size: 13px; padding: 8px 10px; }
.idx-search {
  width: 100%; box-sizing: border-box; background: rgba(6,10,26,0.8);
  border: 1px solid var(--line); border-radius: 9px; color: var(--ink);
  padding: 8px 10px; font-size: 13px; font-family: inherit; outline: none;
  margin-bottom: 6px;
}
.idx-task-row {
  display: flex; flex-direction: column; gap: 1px; width: 100%;
  background: none; border: none; color: var(--ink); cursor: pointer;
  padding: 8px 10px; border-radius: 9px; text-align: left; font-family: inherit;
}
.idx-task-row:hover { background: rgba(150,175,255,0.10); }
.idx-task-title { font-size: 13.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.idx-task-title.done { color: var(--ink-dim); text-decoration: line-through; opacity: 0.7; }
.idx-task-title.q1 { color: var(--warn); }
.idx-task-title.q3 { color: var(--ink-dim); }
.idx-task-title.q1.overdue { color: #ff5a3c; }
.idx-task-title.q3.overdue { color: #d98f3c; }
.idx-task-sub { font-size: 11px; color: var(--ink-dim); opacity: 0.75; }
.idx-badge { font-size: 11.5px; flex-shrink: 0; letter-spacing: 0.02em; }
.idx-badge.q1 { color: var(--warn); }
.idx-badge.q3 { color: var(--ink-dim); }
.idx-badge.idle { color: var(--ink-dim); opacity: 0.65; font-style: italic; }
.idx-legend {
  font-size: 10.5px; color: var(--ink-dim); opacity: 0.8;
  padding: 7px 10px 3px; border-top: 1px solid var(--line); margin-top: 4px;
}

.bottom-right { right: 18px; bottom: 18px; display: flex; gap: 10px; align-items: center; }
.save-note { font-size: 11.5px; color: var(--ink-dim); }
.bottom-right .ghost-btn.sm { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

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
.f-check {
  display: flex; gap: 9px; align-items: baseline; margin: 12px 0 2px;
  font-size: 12.5px; color: var(--ink-dim); cursor: pointer; line-height: 1.45;
}
.f-check input { flex-shrink: 0; accent-color: var(--accent); cursor: pointer; }
.f-check b { color: var(--warn); font-weight: 500; }
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
