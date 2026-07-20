import { createClient } from "@supabase/supabase-js";

// Not secrets — the anon key is meant to ship in the client bundle. Actual
// access control is the Postgres row-level-security policy on `galaxies`
// (see the plan/SQL migration), which restricts every row to its own
// auth.uid(). Missing values just mean cloud sync is silently unavailable
// (falls back to localStorage-only), not a crash — useful for anyone
// building this without a Supabase project configured yet.
const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = url && anonKey ? createClient(url, anonKey) : null;

// Returns whatever raw JSON value is stored in this user's row, or null if
// they have no row yet (first-time sync) or the client isn't configured.
// This is a dumb transport layer — it doesn't know or care about the app's
// state shape (bare projects array vs. the richer {projects, originOptions,
// disciplineOptions} object); Orrery.jsx's normalizeCloudState is what
// interprets whichever shape comes back.
export async function loadCloudState(userId) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from("galaxies")
    .select("projects")
    .eq("user_id", userId)
    .maybeSingle();
  if (error) throw error;
  return data ? data.projects : null;
}

// `state` is the whole app-state object (or, historically, a bare projects
// array) — the DB column is still literally named `projects` for backwards
// compatibility, but it now holds the full {projects, originOptions,
// disciplineOptions} object rather than just the projects array.
export async function saveCloudState(userId, state) {
  if (!supabase) return;
  const { error } = await supabase
    .from("galaxies")
    .upsert({ user_id: userId, projects: state, updated_at: new Date().toISOString() });
  if (error) throw error;
}
