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

// Returns the saved projects array, or null if this user has no row yet
// (first-time sync) or the client isn't configured.
export async function loadCloudProjects(userId) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from("galaxies")
    .select("projects")
    .eq("user_id", userId)
    .maybeSingle();
  if (error) throw error;
  return data ? data.projects : null;
}

export async function saveCloudProjects(userId, projects) {
  if (!supabase) return;
  const { error } = await supabase
    .from("galaxies")
    .upsert({ user_id: userId, projects, updated_at: new Date().toISOString() });
  if (error) throw error;
}
