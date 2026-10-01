// Copyright 2026 Candace Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The report's "CSF architecture" section, for tasks mined from a
//! CSF-instrumented repository (their task.json carries `csf` and
//! `csf_guards`; see src/csf).
//!
//! The section is rendered here as static HTML with an inline SVG chart and
//! inserted before `</main>`, so it needs no chart library and nothing in
//! the page template. Its figure and table numbers continue the page's own
//! (report/figures.rs), and each caption carries what is plotted, the unit,
//! n, how it was computed and a data-derived takeaway. Without CSF data the
//! page is unchanged.

use super::figures::{set_label, Figure};
use super::load::{Split, Task};
use html_escape::encode_text as esc;
use rrsi_mine::csf::guard::GuardVerdict;
use rrsi_mine::csf::map::TaskCsf;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// What task.json says about CSF for one task.
#[derive(Clone, Debug, Default)]
pub struct TaskInfo {
    pub csf: Option<TaskCsf>,
    pub guards: Option<Vec<GuardVerdict>>,
}

/// The CSF fields of every `<sha12>/task.json` (or retry.json) under `dir`,
/// by directory name. Tasks without either field are left out.
pub fn load(dir: &Path) -> BTreeMap<String, TaskInfo> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let d = e.path();
        let meta = ["task.json", "retry.json"].iter().map(|f| d.join(f)).find(|p| p.is_file());
        let Some(v) = meta.and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { continue };
        let info = TaskInfo {
            csf: v.get("csf").and_then(|c| serde_json::from_value(c.clone()).ok()),
            guards: v.get("csf_guards").and_then(|g| serde_json::from_value(g.clone()).ok()),
        };
        if info.csf.is_some() || info.guards.is_some() {
            out.insert(e.file_name().to_string_lossy().into_owned(), info);
        }
    }
    out
}

fn info<'a>(csf: &'a BTreeMap<String, TaskInfo>, t: &Task) -> Option<&'a TaskInfo> {
    csf.get(&t.sha12)
}

/// Use each task's CSF components as its report area when its fix maps to
/// any (`csf:a+b`); other tasks keep the path-based area. Returns how many
/// tasks changed.
pub fn apply_areas(tasks: &mut [Task], csf: &BTreeMap<String, TaskInfo>) -> usize {
    let mut n = 0;
    for t in tasks.iter_mut() {
        if let Some(c) = csf.get(&t.sha12).and_then(|i| i.csf.as_ref()).filter(|c| !c.components.is_empty()) {
            t.area = format!("csf:{}", c.components.join("+"));
            n += 1;
        }
    }
    n
}

const SETS: [Split; 3] = [Split::Evolve, Split::Heldout, Split::Excluded];

fn set_color(s: Split) -> &'static str {
    match s {
        Split::Evolve => "var(--evolve)",
        Split::Heldout => "var(--heldout)",
        Split::Excluded => "var(--excluded)",
    }
}

fn pct(k: usize, n: usize) -> String {
    if n == 0 { "0%".into() } else { format!("{:.0}%", 100.0 * k as f64 / n as f64) }
}

/// A numbered caption, in the page's own caption layout.
struct Cap<'a> {
    kind: &'a str,
    number: usize,
    title: &'a str,
    what: &'a str,
    unit: &'a str,
    n: &'a str,
    how: &'a str,
    takeaway: &'a str,
}

fn caption(c: Cap) -> String {
    let Cap { kind, number, title, what, unit, n, how, takeaway } = c;
    format!("<figcaption><span class=\"fn\">{kind} {number}. </span><span class=\"ft\">{}. </span>{} Unit: {}. n: {}. \
             Computed: {}.<span class=\"tk\"><b>Takeaway: </b>{}</span></figcaption>",
            esc(title), esc(what), esc(unit), esc(n), esc(how), esc(takeaway))
}

/// Horizontal stacked bars: one row per label, one segment per set.
fn bar_chart(rows: &[(String, [usize; 3])], x_title: &str, y_title: &str) -> String {
    let (left, right, top, row_h) = (230.0, 24.0, 12.0, 26.0);
    let width = 720.0;
    let plot_w = width - left - right;
    let max = rows.iter().map(|(_, c)| c.iter().sum::<usize>()).max().unwrap_or(0).max(1);
    let plot_h = row_h * rows.len() as f64;
    let height = top + plot_h + 78.0;
    let x = |v: f64| left + plot_w * v / max as f64;
    let mut s = String::new();
    let _ = write!(s, "<svg class=\"csf-chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" \
                        aria-label=\"{}\" style=\"width:100%;max-width:{width}px;height:auto\">", esc(y_title));
    let step = ((max as f64 / 6.0).ceil() as usize).max(1);
    for v in (0..=max).step_by(step) {
        let xv = x(v as f64);
        let _ = write!(s, "<line x1=\"{xv}\" x2=\"{xv}\" y1=\"{top}\" y2=\"{}\" style=\"stroke:var(--border)\"/>\
                           <text x=\"{xv}\" y=\"{}\" text-anchor=\"middle\" style=\"fill:var(--ink-2);font-size:11px\">{v}</text>",
                       top + plot_h, top + plot_h + 14.0);
    }
    for (i, (label, counts)) in rows.iter().enumerate() {
        let y = top + row_h * i as f64;
        let _ = write!(s, "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" style=\"fill:var(--ink);font-size:12px\">{}</text>",
                       left - 8.0, y + row_h * 0.62, esc(label));
        let mut acc = 0usize;
        for (set, c) in SETS.iter().zip(counts) {
            if *c == 0 {
                continue;
            }
            let (x0, x1) = (x(acc as f64), x((acc + c) as f64));
            let _ = write!(s, "<rect x=\"{x0}\" y=\"{}\" width=\"{}\" height=\"{}\" style=\"fill:{}\">\
                               <title>{}: {} {c} task(s)</title></rect>",
                           y + 4.0, (x1 - x0).max(1.0), row_h - 8.0, set_color(*set), esc(label), set_label(*set));
            acc += c;
        }
        let _ = write!(s, "<text x=\"{}\" y=\"{}\" style=\"fill:var(--ink-2);font-size:11px\">{acc}</text>",
                       x(acc as f64) + 4.0, y + row_h * 0.62);
    }
    let _ = write!(s, "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" style=\"fill:var(--ink);font-size:12px\">{}</text>",
                   left + plot_w / 2.0, top + plot_h + 32.0, esc(x_title));
    let _ = write!(s, "<text x=\"12\" y=\"{}\" style=\"fill:var(--ink);font-size:12px\" \
                        transform=\"rotate(-90 12 {})\" text-anchor=\"middle\">{}</text>",
                   top + plot_h / 2.0, top + plot_h / 2.0, esc(y_title));
    let mut lx = left;
    for set in SETS {
        let _ = write!(s, "<rect x=\"{lx}\" y=\"{}\" width=\"10\" height=\"10\" style=\"fill:{}\"/>\
                           <text x=\"{}\" y=\"{}\" style=\"fill:var(--ink-2);font-size:11px\">{}</text>",
                       top + plot_h + 50.0, set_color(set), lx + 14.0, top + plot_h + 59.0, esc(set_label(set)));
        lx += 160.0;
    }
    s.push_str("</svg>");
    s
}

/// The section, or `None` when no task carries CSF data. `figures` are the
/// page's own, whose numbering this section continues.
pub fn section(tasks: &[Task], csf: &BTreeMap<String, TaskInfo>, figures: &[Figure]) -> Option<String> {
    // A task mined from a CSF repository without a model (no csfc at the
    // time) has a `csf` field with no models: it says nothing about
    // components, so it is counted apart rather than as "touches none".
    let csf_of = |t: &Task| info(csf, t).and_then(|i| i.csf.as_ref());
    let with: Vec<&Task> = tasks.iter().filter(|t| csf_of(t).is_some_and(|c| !c.models.is_empty())).collect();
    let no_model = tasks.iter().filter(|t| csf_of(t).is_some_and(|c| c.models.is_empty())).count();
    let guarded: Vec<&Task> = tasks.iter().filter(|t| info(csf, t).is_some_and(|i| i.guards.is_some())).collect();
    if with.is_empty() && guarded.is_empty() && no_model == 0 {
        return None;
    }
    let next = |kind: &str| figures.iter().filter(|f| f.caption.kind == kind)
        .map(|f| f.caption.number).max().unwrap_or(0) + 1;
    let (fig_n, tab_n) = (next("Figure"), next("Table"));
    let total = tasks.len();
    let mapped: Vec<&Task> = with.iter().copied()
        .filter(|t| info(csf, t).and_then(|i| i.csf.as_ref()).is_some_and(|c| !c.components.is_empty())).collect();
    let exam_ready = tasks.iter().filter(|t| t.split != Split::Excluded).count();
    let exam_mapped = mapped.iter().filter(|t| t.split != Split::Excluded).count();
    let model = with.iter().find_map(|t| info(csf, t).and_then(|i| i.csf.as_ref()))
        .map(|c| (c.models.join(", "), c.model_rev.clone().unwrap_or_default())).unwrap_or_default();

    // Figure: components touched, per set.
    let mut per: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for t in &with {
        let c = info(csf, t).and_then(|i| i.csf.as_ref()).expect("filtered");
        let keys = if c.components.is_empty() { vec!["(no declared component)".to_string()] } else { c.components.clone() };
        for k in keys {
            per.entry(k).or_default()[SETS.iter().position(|s| *s == t.split).unwrap_or(2)] += 1;
        }
    }
    let mut rows: Vec<(String, [usize; 3])> = per.into_iter().collect();
    rows.sort_by(|a, b| {
        let none = |r: &(String, [usize; 3])| r.0.starts_with('(');
        none(a).cmp(&none(b)).then(b.1.iter().sum::<usize>().cmp(&a.1.iter().sum::<usize>())).then(a.0.cmp(&b.0))
    });
    let top = rows.iter().find(|r| !r.0.starts_with('('));

    let mut h = String::new();
    h.push_str("<section id=\"sec-csf\">\n<h2>CSF architecture</h2>\n");
    h.push_str("<p class=\"q\">Which parts of the declared CSF architecture do the tasks exercise, and do the reference fixes pass CSF's own gates?</p>\n");
    let _ = writeln!(h, "<p class=\"how\"><b>How to read this:</b> the repository declares its architecture in CSF \
        (model <code>{}</code>{}). Each task's fix is mapped to the declared component whose source path is the \
        longest prefix of each changed file; CSF's compiler <code>csfc</code> then checks the fix's own commit tree.</p>",
        esc(&model.0), if model.1.is_empty() { String::new() } else { format!(", read at <code>{}</code>", esc(&model.1)) });

    if no_model > 0 {
        let _ = writeln!(h, "<p class=\"banner\">{no_model} task(s) come from a CSF repository but were mined without \
            an architecture model (no csfc, or csfc rejected the model), so their components are unknown. \
            Re-run <code>rrsi-mine csf annotate --csfc PATH</code> to add them.</p>");
    }
    if !with.is_empty() {
        let top_note = match top {
            Some((name, c)) => format!("{} is touched most: {} task(s).", name, c.iter().sum::<usize>()),
            None => "No task's fix touches a declared component.".into(),
        };
        let _ = writeln!(h, "<figure class=\"fig\" data-fig=\"csf-components\">{}{}</figure>",
            bar_chart(&rows, "Tasks (count)", "CSF component"),
            caption(Cap {
                kind: "Figure",
                number: fig_n,
                title: "Declared CSF components the reference fixes touch, per set",
                what: "One bar per declared component (plus tasks touching none), stacked by set; a task touching two components counts in both bars.",
                unit: "tasks (count)",
                n: &format!("{} of {total} tasks carry a CSF component map; {} of them touch at least one component",
                            with.len(), mapped.len()),
                how: "each changed non-test, non-generated Go file of the fix is assigned the component whose declared source is its longest path prefix (src/csf/map.rs)",
                takeaway: &format!("{top_note} {exam_mapped} of {exam_ready} exam-ready tasks ({}) touch a declared component, \
                                    so the rest of the exam exercises code the architecture model does not describe.",
                                   pct(exam_mapped, exam_ready)),
            }));
    }

    if !guarded.is_empty() {
        let mut gates: BTreeMap<String, [usize; 5]> = BTreeMap::new();
        let mut failing: Vec<(&Task, &GuardVerdict)> = Vec::new();
        for t in &guarded {
            for v in info(csf, t).and_then(|i| i.guards.as_ref()).into_iter().flatten() {
                let row = gates.entry(v.gate.clone()).or_default();
                let col = ["pass", "fail", "error", "skipped"].iter().position(|s| *s == v.status).unwrap_or(2);
                row[col] += 1;
                row[4] += v.required_of_agent.unwrap_or(false) as usize;
                if v.status == "fail" {
                    failing.push((t, v));
                }
            }
        }
        let num = "style=\"text-align:right\"";
        let mut table = format!("<div class=\"tablewrap\"><table><thead><tr><th>Gate</th><th {num}>pass</th>\
            <th {num}>fail</th><th {num}>error</th><th {num}>skipped</th><th {num}>required of the agent</th></tr></thead><tbody>");
        for (g, c) in &gates {
            let _ = write!(table, "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
                <td class=\"num\">{}</td><td class=\"num\">{}</td></tr>", esc(g), c[0], c[1], c[2], c[3], c[4]);
        }
        table.push_str("</tbody></table></div>");
        let n_guarded = guarded.len();
        let skipped_all = gates.values().all(|c| c[3] == c.iter().take(4).sum::<usize>());
        let fails = failing.iter().map(|(t, _)| t.sha12.as_str()).collect::<std::collections::BTreeSet<_>>().len();
        let checked = guarded.iter().filter(|t| info(csf, t).and_then(|i| i.guards.as_ref())
            .is_some_and(|g| g.iter().any(|v| v.status != "skipped"))).count();
        let skip_reason = guarded.iter().filter_map(|t| info(csf, t).and_then(|i| i.guards.as_ref()))
            .flatten().find(|v| v.status == "skipped").map(|v| v.reason.clone()).unwrap_or_default();
        let takeaway = if skipped_all {
            "Every gate was skipped (no csfc or no grammar when mining), so no task is guarded yet.".to_string()
        } else {
            let unchecked = if checked < n_guarded {
                format!(" The other {} task(s) were not checked (for example: {skip_reason}).", n_guarded - checked)
            } else {
                String::new()
            };
            format!("{fails} of {checked} checked reference fixes ({}) fail at least one gate; \
                     those gates are not required of an agent on those tasks.{unchecked}", pct(fails, checked))
        };
        let _ = writeln!(h, "<figure class=\"tbl\" data-fig=\"csf-guards\">{}{table}</figure>",
            caption(Cap {
                kind: "Table",
                number: tab_n,
                title: "CSF gate outcomes on the reference fixes",
                what: "Per gate, how many tasks' commit trees passed, failed, could not be judged (error) or were not checked (skipped), and on how many the gate is required of an agent.",
                unit: "tasks (count)",
                n: &format!("{n_guarded} of {total} tasks have guard records"),
                how: "csfc check and csfc check-generated run read-only on each task's commit tree during mining (src/csf/guard.rs); a gate is required of the agent only where the reference fix passed it",
                takeaway: &takeaway,
            }));
        if !failing.is_empty() {
            let mut t = String::from("<div class=\"tablewrap\"><table><thead><tr><th>Task</th><th>Set</th><th>Subject</th>\
                <th>Gate</th><th>First diagnostic</th></tr></thead><tbody>");
            for (task, v) in &failing {
                let d = v.diagnostics.first().map(|d| format!("{}:{}:{}: {}: {}", d.file, d.line, d.col, d.code, d.message))
                    .unwrap_or_else(|| v.reason.clone());
                let _ = write!(t, "<tr><td class=\"nw mono\">{}</td><td class=\"nw\">{}</td><td>{}</td><td class=\"nw\">{}</td><td class=\"mono\">{}</td></tr>",
                               esc(&task.sha12), esc(set_label(task.split)), esc(&task.subject), esc(&v.gate), esc(&d));
            }
            t.push_str("</tbody></table></div>");
            let _ = writeln!(h, "<figure class=\"tbl\" data-fig=\"csf-failing\">{}{t}</figure>",
                caption(Cap {
                    kind: "Table",
                    number: tab_n + 1,
                    title: "Tasks whose reference fix fails a CSF gate",
                    what: "One row per failing gate of a task, with csfc's first diagnostic.",
                    unit: "tasks and gates",
                    n: &format!("{} failing gate result(s) on {fails} task(s)", failing.len()),
                    how: "csfc's diagnostics as printed (file:line:col: code: message)",
                    takeaway: &failing_takeaway(&failing),
                }));
        }
    }

    let _ = writeln!(h, "<div class=\"card\"><dl>\
        <dt>What it means</dt><dd>{} of {exam_ready} exam-ready tasks ({}) change code that the CSF architecture model declares; for those, the report can name the component (service, manager, library, adapter, gateway or resource) instead of a folder.</dd>\
        <dt>Why it matters</dt><dd>An exam that never touches declared components cannot show whether an agent respects the architecture; a gate the reference fix fails would punish an agent for the task author's choice.</dd>\
        <dt>What to do</dt><dd>Declare more of the repository's components in its architecture.csf (or mine a repository whose model covers more of its code) to raise coverage; mine with <code>--csfc</code> so every task records its gate verdicts; pass the gates marked required to the grader (<code>rrsi_mine.csf_guard</code>).</dd>\
        </dl></div>\n</section>", exam_mapped, pct(exam_mapped, exam_ready));
    Some(h)
}

/// The takeaway for the failing-gate table. Projection drift alone is
/// called out: the projections embed the generator version, so a csfc other
/// than the one the commit's CSF pins reports it on an unchanged model.
fn failing_takeaway(failing: &[(&Task, &GuardVerdict)]) -> String {
    let drift_only = failing.iter().all(|(_, v)| !v.diagnostics.is_empty()
        && v.diagnostics.iter().all(|d| d.code == "CSF_GENERATED_DRIFT"));
    if drift_only {
        format!("All {} failures are CSF_GENERATED_DRIFT only: the checked-in projections differ from what this csfc emits. \
                 Unless this csfc is the version the commit's CSF pins, that is compiler drift rather than a defect of the fix; \
                 either way the gate is not required of an agent on these tasks.", failing.len())
    } else {
        "The fix's own tree fails the gate, so an agent cannot fairly be required to pass it on these tasks; the task stays in the exam for its tests.".into()
    }
}

/// `page` with `section` inserted before `</main>` (or `</body>`).
pub fn inject(page: &str, section: Option<&str>) -> String {
    let Some(s) = section else { return page.to_string() };
    for tag in ["</main>", "</body>"] {
        if let Some(i) = page.rfind(tag) {
            return format!("{}{s}{}", &page[..i], &page[i..]);
        }
    }
    format!("{page}{s}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::load::load_tasks;
    use serde_json::json;

    fn write(root: &Path, name: &str, valid: bool, extra: serde_json::Value) {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let mut v = json!({"sha": format!("{name}{}", "0".repeat(28)), "parent": "1".repeat(40),
            "subject": "orders: <b>place</b> & pay", "body": "", "module_root": "svc", "packages": ["./services/orders"],
            "src_files": ["svc/services/orders/place.go"], "test_files": [], "src_churn": 3,
            "valid": valid, "reason": if valid { "ok" } else { "tests already pass on parent" }, "seconds": 1.0,
            "parent_outcome": "BuildFail", "commit_outcome": "Pass"});
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        std::fs::write(d.join("task.json"), v.to_string()).unwrap();
    }

    fn csf(components: &[&str]) -> serde_json::Value {
        json!({"models": ["shop"], "model_rev": "abc", "components": components, "kinds": ["service"],
               "state": if components.is_empty() { serde_json::Value::Null } else { json!("existing") },
               "files_mapped": components.len(), "files_total": 1})
    }

    fn guard(gate: &str, status: &str) -> serde_json::Value {
        json!({"gate": gate, "model": "svc/csf/architecture/architecture.csf", "status": status,
               "diagnostics": if status == "fail" { json!([{"file": "a.csf", "line": 3, "col": 1,
                   "code": "CSF_PATH", "message": "symlink <x>"}]) } else { json!([]) },
               "required_of_agent": status == "pass"})
    }

    #[test]
    fn no_csf_data_leaves_the_page_unchanged() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "aaaaaaaaaaaa", true, json!({}));
        let tasks = load_tasks(d.path()).unwrap();
        let info = load(d.path());
        assert!(info.is_empty());
        assert_eq!(section(&tasks, &info, &[]), None);
        assert_eq!(inject("<main>x</main>", None), "<main>x</main>");
    }

    #[test]
    fn the_section_numbers_captions_counts_and_escapes() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "aaaaaaaaaaaa", true, json!({"csf": csf(&["orders"]),
            "csf_guards": [guard("csfc check", "pass"), guard("csfc check-generated", "fail")]}));
        write(d.path(), "bbbbbbbbbbbb", true, json!({"csf": csf(&[]),
            "csf_guards": [guard("csfc check", "pass"), guard("csfc check-generated", "pass")]}));
        write(d.path(), "cccccccccccc", false, json!({"csf": csf(&["orders", "money"])}));
        let mut tasks = load_tasks(d.path()).unwrap();
        crate::report::split::assign(&mut tasks, 1);
        let info = load(d.path());
        assert_eq!(info.len(), 3);
        let figs: Vec<Figure> = Vec::new();
        let s = section(&tasks, &info, &figs).unwrap();
        assert!(s.contains("Figure 1. </span>"), "numbering starts after the page's own");
        assert!(s.contains("Table 1. </span>") && s.contains("Table 2. </span>"));
        assert!(s.contains("Unit: tasks (count)") && s.contains("n: 3 of 3 tasks carry a CSF component map; 2 of them touch"));
        assert!(s.contains("Tasks (count)</text>") && s.contains(">CSF component</text>"), "axis titles");
        assert!(s.contains(">orders</text>") && s.contains(">money</text>") && s.contains("(no declared component)"));
        assert!(s.contains("1 of 2 checked reference fixes (50%) fail at least one gate"), "{s}");
        assert!(s.contains("so an agent cannot fairly be required"), "a CSF_PATH failure is the fix's own");
        assert!(s.contains("orders: &lt;b&gt;place&lt;/b&gt; &amp; pay"), "subjects are escaped");
        assert!(s.contains("symlink &lt;x&gt;") && !s.contains("<x>"));
        assert!(s.contains("What it means") && s.contains("Why it matters") && s.contains("What to do"));
        // One exam-ready task maps (aaaa), bbbb maps to none, cccc is not exam-ready.
        assert!(s.contains("1 of 2 exam-ready tasks (50%)"), "{s}");
        let page = inject("<main><p>a</p></main></body>", Some(&s));
        assert!(page.find("sec-csf").unwrap() < page.find("</main>").unwrap());
    }

    #[test]
    fn drift_only_failures_are_called_out() {
        let d = tempfile::tempdir().unwrap();
        let drift = json!({"gate": "csfc check-generated", "model": null, "status": "fail",
            "diagnostics": [{"file": "g/review_cgen.md", "line": 1, "col": 1, "code": "CSF_GENERATED_DRIFT",
                             "message": "run csfc emit"}], "required_of_agent": false});
        write(d.path(), "aaaaaaaaaaaa", true, json!({"csf": csf(&[]), "csf_guards": [drift]}));
        let tasks = load_tasks(d.path()).unwrap();
        let s = section(&tasks, &load(d.path()), &[]).unwrap();
        assert!(s.contains("All 1 failures are CSF_GENERATED_DRIFT only"), "{s}");
    }

    #[test]
    fn tasks_mined_without_a_model_are_counted_apart() {
        let d = tempfile::tempdir().unwrap();
        let mut no_model = csf(&[]);
        no_model["models"] = json!([]);
        write(d.path(), "aaaaaaaaaaaa", true, json!({"csf": no_model}));
        let tasks = load_tasks(d.path()).unwrap();
        let s = section(&tasks, &load(d.path()), &[]).unwrap();
        assert!(s.contains("1 task(s) come from a CSF repository but were mined without"), "{s}");
        assert!(!s.contains("csf-components"), "no component chart without a model");
    }

    #[test]
    fn all_skipped_guards_say_so() {
        let d = tempfile::tempdir().unwrap();
        let skipped = json!({"gate": "csfc check", "model": null, "status": "skipped", "reason": "no csfc",
                             "required_of_agent": false});
        write(d.path(), "aaaaaaaaaaaa", true, json!({"csf": csf(&[]), "csf_guards": [skipped]}));
        let tasks = load_tasks(d.path()).unwrap();
        let s = section(&tasks, &load(d.path()), &[]).unwrap();
        assert!(s.contains("Every gate was skipped"), "{s}");
        assert!(!s.contains("csf-failing"));
    }

    #[test]
    fn areas_become_components_only_where_a_fix_maps() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "aaaaaaaaaaaa", true, json!({"csf": csf(&["money", "orders"])}));
        write(d.path(), "bbbbbbbbbbbb", true, json!({"csf": csf(&[])}));
        let mut tasks = load_tasks(d.path()).unwrap();
        let before = tasks[1].area.clone();
        assert_eq!(apply_areas(&mut tasks, &load(d.path())), 1);
        assert_eq!(tasks[0].area, "csf:money+orders");
        assert_eq!(tasks[1].area, before);
    }
}
