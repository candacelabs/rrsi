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


//! Deterministic text analysis behind the fairness stages: what API the
//! hidden tests need (`required_api`), whether a problem statement leaks the
//! real fix (`leak_check`), and which assertions pin details a statement
//! would have to spell out (`specificity`). Everything works on unified diffs
//! of gofmt'd Go and on `go test` logs; nothing here runs a process.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// One declaration found in Go source or in the added lines of a patch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decl {
    pub name: String,
    /// func, method, type, const, var or field.
    pub kind: String,
    /// The declaring line, trimmed, without a one-line body.
    pub signature: String,
    pub file: String,
}

/// A symbol the hidden tests need that the parent tree lacks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Required {
    pub symbol: String,
    pub kind: String,
    /// The declaring line from src.patch, or "" when only the log names it.
    pub signature: String,
}

/// One file of a unified diff: its new path and, per hunk, the text after
/// the second `@@` plus each line with its marker (' ', '+' or '-').
pub struct FileDiff {
    pub path: String,
    pub hunks: Vec<(String, Vec<(char, String)>)>,
}

pub fn parse_diff(patch: &str) -> Vec<FileDiff> {
    let mut files: Vec<FileDiff> = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            let path = rest.rsplit_once(" b/").map(|(_, b)| b).unwrap_or(rest).to_string();
            files.push(FileDiff { path, hunks: Vec::new() });
        } else if line.starts_with("+++ ") || line.starts_with("--- ") || line.starts_with("index ") {
            continue;
        } else if let Some(rest) = line.strip_prefix("@@") {
            let ctx = rest.split_once("@@").map(|(_, c)| c.trim()).unwrap_or("").to_string();
            if let Some(f) = files.last_mut() {
                f.hunks.push((ctx, Vec::new()));
            }
        } else if let Some((_, lines)) = files.last_mut().and_then(|f| f.hunks.last_mut()) {
            let mut chars = line.chars();
            match chars.next() {
                Some(m @ (' ' | '+' | '-')) => lines.push((m, chars.as_str().to_string())),
                None => lines.push((' ', String::new())),
                _ => {}
            }
        }
    }
    files
}

/// The part of a patch touching files directly in one of `dirs`.
pub fn patch_in_dirs(patch: &str, dirs: &[String]) -> String {
    let mut out = String::new();
    let mut keep = false;
    for line in patch.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            let path = rest.trim_end().rsplit_once(" b/").map(|(_, b)| b).unwrap_or("");
            keep = dirs.iter().any(|d| crate::dir_of(path) == *d);
        }
        if keep {
            out.push_str(line);
        }
    }
    out
}

/// The added lines of a patch (without the `+`), in order.
pub fn added_lines(patch: &str) -> Vec<String> {
    parse_diff(patch).into_iter().flat_map(|f| f.hunks).flat_map(|(_, l)| l)
        .filter(|(m, _)| *m == '+').map(|(_, l)| l).collect()
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The identifier at the start of `s`, if any.
fn leading_ident(s: &str) -> Option<&str> {
    let end = s.find(|c: char| !is_ident_char(c)).unwrap_or(s.len());
    let id = &s[..end];
    (!id.is_empty() && !id.starts_with(|c: char| c.is_ascii_digit())).then_some(id)
}

pub fn is_exported(name: &str) -> bool {
    name.starts_with(|c: char| c.is_uppercase())
}

/// Whether `word` occurs in `text` as a whole identifier.
pub fn has_word(text: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    text.match_indices(word).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + word.len()..].chars().next();
        !before.is_some_and(is_ident_char) && !after.is_some_and(is_ident_char)
    })
}

/// A declaring line without a one-line body: `func F() int { return 1 }`
/// becomes `func F() int`, `type T struct {` becomes `type T struct`.
pub fn strip_body(line: &str) -> String {
    let t = line.trim();
    let (mut paren, mut bracket) = (0i32, 0i32);
    for (i, c) in t.char_indices() {
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' if paren == 0 && bracket == 0 && i > 0 && t[..i].ends_with(' ') => {
                return t[..i].trim_end().to_string();
            }
            _ => {}
        }
    }
    t.to_string()
}

#[derive(Clone, Copy, PartialEq)]
enum Block { None, Struct, Interface, Const, Var, Type }

fn block_of(top: &str) -> Block {
    let t = top.trim_end();
    if t == "const (" { Block::Const }
    else if t == "var (" { Block::Var }
    else if t == "type (" { Block::Type }
    else if t.starts_with("type ") && t.ends_with("struct {") { Block::Struct }
    else if t.starts_with("type ") && t.ends_with("interface {") { Block::Interface }
    else { Block::None }
}

/// Declarations on the selected lines of one gofmt'd stretch of Go. `ctx`
/// is the enclosing top-level line (a hunk header's text). Column-0 lines
/// are top-level; one-tab lines belong to the last top-level block opener.
pub fn scan_decls<'a>(ctx: &str, lines: impl IntoIterator<Item = (bool, &'a str)>, file: &str) -> Vec<Decl> {
    let mut out = Vec::new();
    let mut block = block_of(ctx);
    let mut push = |name: &str, kind: &str, line: &str| out.push(Decl {
        name: name.to_string(), kind: kind.to_string(), signature: strip_body(line), file: file.to_string(),
    });
    for (selected, line) in lines {
        if line.trim().is_empty() || line.trim_start().starts_with("//") {
            continue;
        }
        if !line.starts_with(char::is_whitespace) {
            block = block_of(line);
            if !selected {
                continue;
            }
            if let Some(rest) = line.strip_prefix("func ") {
                if let Some(r) = rest.strip_prefix('(') {
                    // Method: skip the receiver's balanced parentheses.
                    let mut depth = 1;
                    let close = r.char_indices().find(|&(_, c)| {
                        depth += match c { '(' => 1, ')' => -1, _ => 0 };
                        depth == 0
                    });
                    if let Some(name) = close.and_then(|(i, _)| leading_ident(r[i + 1..].trim_start())) {
                        push(name, "method", line);
                    }
                } else if let Some(name) = leading_ident(rest) {
                    push(name, "func", line);
                }
            } else if let Some(rest) = line.strip_prefix("type ") {
                if let Some(name) = leading_ident(rest) {
                    push(name, "type", line);
                }
            } else {
                for kw in ["const", "var"] {
                    if let Some(rest) = line.strip_prefix(kw).and_then(|r| r.strip_prefix(' ')) {
                        for n in rest.split('=').next().unwrap_or("").split(',') {
                            if let Some(name) = leading_ident(n.trim()) {
                                push(name, kw, line);
                            }
                        }
                    }
                }
            }
            continue;
        }
        // One tab deep: a member of the current top-level block.
        let Some(member) = line.strip_prefix('\t').filter(|m| !m.starts_with(char::is_whitespace)) else {
            continue;
        };
        if !selected || member.starts_with('}') || member.starts_with(')') {
            continue;
        }
        let first = leading_ident(member);
        match (block, first) {
            (Block::Const | Block::Var, Some(_)) => {
                let kind = if block == Block::Const { "const" } else { "var" };
                let names = member.split('=').next().unwrap_or(member);
                // `A, B = 1, 2` or `A Type = x` or a bare `A` (iota).
                let names = if names.contains(',') { names.to_string() }
                            else { names.split_whitespace().next().unwrap_or("").to_string() };
                for n in names.split(',') {
                    if let Some(name) = leading_ident(n.trim()) {
                        push(name, kind, member);
                    }
                }
            }
            (Block::Type, Some(name)) => push(name, "type", member),
            (Block::Struct, Some(_)) => {
                // `A, B int` declares fields; a lone type name is an embedding.
                let words: Vec<&str> = member.split("//").next().unwrap_or("").split_whitespace().collect();
                if words.len() >= 2 {
                    let commas = words.iter().take_while(|w| w.ends_with(',')).count();
                    for n in &words[..=commas.min(words.len() - 1)] {
                        if let Some(name) = leading_ident(n.trim_end_matches(',')) {
                            push(name, "field", member);
                        }
                    }
                }
            }
            (Block::Interface, Some(name)) if member[name.len()..].starts_with('(') => {
                push(name, "method", member);
            }
            _ => {}
        }
    }
    dedup(out)
}

fn dedup(decls: Vec<Decl>) -> Vec<Decl> {
    let mut seen = HashSet::new();
    decls.into_iter().filter(|d| seen.insert((d.name.clone(), d.signature.clone()))).collect()
}

/// Declarations on the added lines of non-test Go files of a patch.
pub fn added_decls(patch: &str) -> Vec<Decl> {
    let mut out = Vec::new();
    for f in parse_diff(patch) {
        if !f.path.ends_with(".go") || f.path.ends_with("_test.go") {
            continue;
        }
        for (ctx, lines) in &f.hunks {
            let kept = lines.iter().filter(|(m, _)| *m != '-').map(|(m, l)| (*m == '+', l.as_str()));
            out.extend(scan_decls(ctx, kept, &f.path));
        }
    }
    dedup(out)
}

/// Exported declarations of one whole Go source file.
pub fn exported_api(source: &str, file: &str) -> Vec<Decl> {
    scan_decls("", source.lines().map(|l| (true, l)), file).into_iter()
        .filter(|d| is_exported(&d.name)).collect()
}

/// Symbols a `go test` log says are missing: `undefined: X`, `x.Y undefined
/// (type T has no field or method Y)`, `unknown field Y in struct literal`,
/// and packages of the module itself that do not exist yet.
pub fn log_symbols(log: &str, own_module: Option<&str>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut add = |s: &str, k: &str| {
        if !s.is_empty() && !out.iter().any(|(o, _)| o == s) {
            out.push((s.to_string(), k.to_string()));
        }
    };
    let ident_after = |line: &str, marker: &str| -> Option<String> {
        let rest = &line[line.find(marker)? + marker.len()..];
        let end = rest.find(|c: char| !(is_ident_char(c) || c == '.')).unwrap_or(rest.len());
        let q = &rest[..end];
        q.rsplit('.').next().map(str::to_string)
    };
    for line in log.lines() {
        if let Some(s) = ident_after(line, "has no field or method ") {
            add(&s, "field or method");
        } else if let Some(s) = ident_after(line, "undefined: ") {
            add(&s, "identifier");
        } else if let Some(s) = ident_after(line, "unknown field ") {
            add(&s, "field");
        } else if let (Some(m), Some(i)) = (own_module, line.find("providing package ")) {
            let pkg = line[i + "providing package ".len()..].split([':', ' ']).next().unwrap_or("");
            if let Some(rel) = pkg.strip_prefix(m).and_then(|r| r.strip_prefix('/')) {
                add(rel, "package");
            }
        }
    }
    out
}

/// The public (and package-internal) API the hidden tests need: symbols the
/// parent's log reports missing, plus declarations src.patch adds in
/// `package_dirs` that tests.patch's added lines reference.
pub fn required_api(parent_log: &str, own_module: Option<&str>, src_patch: &str,
                    tests_patch: &str, package_dirs: &[String]) -> Vec<Required> {
    let decls: Vec<Decl> = added_decls(src_patch).into_iter()
        .filter(|d| package_dirs.iter().any(|p| crate::dir_of(&d.file) == *p)).collect();
    let test_text = added_lines(tests_patch).join("\n");
    let mut out: Vec<Required> = Vec::new();
    let mut add = |r: Required| {
        if !out.iter().any(|o| o.symbol == r.symbol && o.signature == r.signature) {
            out.push(r);
        }
    };
    for (sym, kind) in log_symbols(parent_log, own_module) {
        let found: Vec<&Decl> = decls.iter().filter(|d| d.name == sym).collect();
        if found.is_empty() {
            add(Required { symbol: sym, kind, signature: String::new() });
        }
        for d in found {
            add(Required { symbol: d.name.clone(), kind: d.kind.clone(), signature: d.signature.clone() });
        }
    }
    for d in &decls {
        if has_word(&test_text, &d.name) {
            add(Required { symbol: d.name.clone(), kind: d.kind.clone(), signature: d.signature.clone() });
        }
    }
    out
}

/// Lines that are only a comment or only a string (an import spec).
fn is_code(line: &str) -> bool {
    let t = line.trim();
    !(t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
      || (t.starts_with('"') && t.ends_with('"') && t.matches('"').count() == 2))
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Go-ish tokens: identifiers and numbers, whole string literals, and single
/// punctuation characters. Markdown emphasis and code marks are dropped.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() || matches!(c, '`' | '*' | '#') {
            i += 1;
        } else if is_ident_char(c) {
            let s = i;
            while i < cs.len() && is_ident_char(cs[i]) { i += 1; }
            out.push(cs[s..i].iter().collect());
        } else if c == '"' {
            let s = i;
            i += 1;
            while i < cs.len() && cs[i] != '"' && cs[i] != '\n' {
                if cs[i] == '\\' { i += 1; }
                i += 1;
            }
            i = (i + 1).min(cs.len());
            out.push(cs[s..i].iter().collect());
        } else {
            out.push(c.to_string());
            i += 1;
        }
    }
    out
}

pub const SHINGLE: usize = 8;
/// A statement may share at most this many 8-token runs with the fix's
/// added code (a qualified call and its arguments can coincide by accident).
pub const MAX_SHINGLE_OVERLAP: usize = 2;
/// Added lines at least this long (trimmed) must never appear verbatim.
pub const MIN_LEAK_LINE: usize = 25;

fn shingles(toks: &[String]) -> HashSet<String> {
    toks.windows(SHINGLE).map(|w| w.join("\u{1}")).collect()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LeakReport {
    /// Added src.patch lines found verbatim (whitespace-normalised).
    pub verbatim: Vec<String>,
    /// Distinct 8-token runs shared with the fix's added code.
    pub shingle_overlap: usize,
}

impl LeakReport {
    pub fn leaks(&self) -> bool {
        !self.verbatim.is_empty() || self.shingle_overlap > MAX_SHINGLE_OVERLAP
    }
}

/// Whether `instruction` reproduces the fix. The required signatures are
/// allowed (a statement must name them exactly); comments are not code.
pub fn leak_check(instruction: &str, src_patch: &str, signatures: &[String]) -> LeakReport {
    let allowed: HashSet<String> = signatures.iter().map(|s| squash(s)).collect();
    // The part of an added line that is not an allowed signature: nothing
    // for a declaration line, the body of a one-line `func ... { body }`.
    let unsigned = |l: &str| -> String {
        let sig = strip_body(l);
        if allowed.contains(&squash(l)) {
            String::new()
        } else if allowed.contains(&squash(&sig)) {
            l.trim()[sig.len()..].to_string()
        } else {
            l.to_string()
        }
    };
    let text = squash(instruction);
    let mut rep = LeakReport::default();
    let mut src_shingles = HashSet::new();
    for f in parse_diff(src_patch) {
        for (_, lines) in &f.hunks {
            let mut run: Vec<String> = Vec::new();
            for (m, l) in lines {
                let rest = if *m == '+' && is_code(l) { unsigned(l) } else { String::new() };
                if rest.trim().is_empty() {
                    src_shingles.extend(shingles(&run));
                    run.clear();
                    continue;
                }
                let q = squash(&rest);
                if q.len() >= MIN_LEAK_LINE && text.contains(&q) && !rep.verbatim.contains(&q) {
                    rep.verbatim.push(q);
                }
                run.extend(tokens(rest.split("//").next().unwrap_or(&rest)));
            }
            src_shingles.extend(shingles(&run));
        }
    }
    rep.shingle_overlap = shingles(&tokens(instruction)).intersection(&src_shingles).count();
    rep
}

/// Phrases a problem statement must never contain: it is an issue, not a
/// description of the grading or of a commit.
pub const FORBIDDEN_MENTIONS: [&str; 7] =
    ["hidden test", "_test.go", "tests.patch", "src.patch", "this commit", "the commit", "the fix is"];

pub fn forbidden_mentions(instruction: &str) -> Vec<String> {
    let low = instruction.to_lowercase();
    FORBIDDEN_MENTIONS.iter().filter(|p| low.contains(*p)).map(|p| p.to_string()).collect()
}

/// An assertion that pins something a statement would have to spell out.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Finding {
    /// message, unexported or call_count.
    pub kind: String,
    pub detail: String,
    pub line: String,
    pub covered: bool,
}

pub const MIN_PINNED_STRING: usize = 20;
const MATCHERS: [&str; 9] = ["Equal(", "ContainSubstring(", "MatchError(", "EqualError(",
                             "HavePrefix(", "HaveSuffix(", "==", "!=", "ErrorContains("];

/// String literals on a line: (content, literal-with-quotes).
fn string_literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let cs: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let q = cs[i];
        if q == '"' || q == '`' {
            let s = i + 1;
            i += 1;
            while i < cs.len() && cs[i] != q {
                if q == '"' && cs[i] == '\\' { i += 1; }
                i += 1;
            }
            if i < cs.len() {
                out.push(cs[s..i].iter().collect());
            }
        }
        i += 1;
    }
    out
}

fn count_words(n: u64) -> &'static [&'static str] {
    match n {
        0 => &["zero", "never", "not ", "without", "no "],
        2 => &["twice", "two"],
        3 => &["three", "thrice"],
        4 => &["four"],
        5 => &["five"],
        _ => &[],
    }
}

/// Over-specific assertions in tests.patch's added lines:
/// - `message`: a string literal of 20+ characters compared with a matcher,
///   unless the tests also use it as input or the parent code already has it
///   (`existing` is the parent's package source; src.patch context and
///   removed lines count as existing too);
/// - `unexported`: a new unexported declaration of src.patch the tests use;
/// - `call_count`: a gomock `.Times(n)` other than the default 1.
///
/// A finding is covered when the instruction names the string, identifier
/// or mocked method (and, for a literal count, the count).
pub fn specificity(tests_patch: &str, src_patch: &str, instruction: &str, existing: &str,
                   package_dirs: &[String]) -> Vec<Finding> {
    let mut out = Vec::new();
    let test_lines: Vec<(char, String)> = parse_diff(tests_patch).into_iter()
        .flat_map(|f| f.hunks).flat_map(|(_, l)| l).collect();
    let old_src: String = parse_diff(src_patch).into_iter().flat_map(|f| f.hunks)
        .flat_map(|(_, l)| l).filter(|(m, _)| *m != '+').map(|(_, l)| l + "\n").collect();
    let low_instr = instruction.to_lowercase();
    let added: Vec<&str> = test_lines.iter().filter(|(m, _)| *m == '+').map(|(_, l)| l.as_str()).collect();
    let mut seen = BTreeSet::new();
    for (idx, line) in added.iter().enumerate() {
        let code = line.trim();
        if code.starts_with("//") {
            continue;
        }
        if MATCHERS.iter().any(|m| code.contains(m)) {
            for lit in string_literals(code) {
                if lit.chars().count() < MIN_PINNED_STRING || !seen.insert(("message", lit.clone())) {
                    continue;
                }
                let as_input = test_lines.iter().any(|(m, l)| *m != '-' && l.contains(&lit)
                    && !MATCHERS.iter().any(|x| l.contains(x)));
                if as_input || existing.contains(&lit) || old_src.contains(&lit) {
                    continue;
                }
                out.push(Finding { kind: "message".into(), covered: instruction.contains(&lit),
                                   detail: lit, line: code.to_string() });
            }
        }
        if let Some(i) = code.find(".Times(") {
            let arg = code[i + 7..].split(')').next().unwrap_or("").trim().to_string();
            if arg != "1" {
                let window = added[idx.saturating_sub(8)..=idx].join("\n");
                let method = window.rfind("EXPECT().").and_then(|j| leading_ident(&window[j + 9..]))
                    .unwrap_or("").to_string();
                let names = !method.is_empty() && low_instr.contains(&method.to_lowercase());
                let count = match arg.parse::<u64>() {
                    Ok(n) => has_word(instruction, &n.to_string())
                        || count_words(n).iter().any(|w| low_instr.contains(w)),
                    Err(_) => true,
                };
                out.push(Finding { kind: "call_count".into(), covered: names && count,
                                   detail: format!("{method}.Times({arg})"), line: code.to_string() });
            }
        }
    }
    let test_text = added.join("\n");
    let mut names = BTreeMap::new();
    for d in added_decls(src_patch) {
        if !is_exported(&d.name) && package_dirs.iter().any(|p| crate::dir_of(&d.file) == *p)
            && has_word(&test_text, &d.name) {
            names.entry(d.name.clone()).or_insert(d);
        }
    }
    for (name, d) in names {
        let line = added.iter().find(|l| has_word(l, &name)).map(|l| l.trim().to_string()).unwrap_or_default();
        out.push(Finding { kind: "unexported".into(), covered: has_word(instruction, &name),
                           detail: format!("{} {}", d.kind, name), line });
    }
    out
}

/// The reviewer's verdict on a problem statement.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Probe {
    pub sufficient: bool,
    pub ambiguities: Vec<serde_json::Value>,
    pub guessed_names: Vec<serde_json::Value>,
}

/// Parse the reviewer's reply: the outermost JSON object in it, which must
/// carry all three fields with the right types.
pub fn parse_probe(reply: &str) -> anyhow::Result<Probe> {
    let (s, e) = (reply.find('{'), reply.rfind('}'));
    match (s, e) {
        (Some(s), Some(e)) if s < e => Ok(serde_json::from_str(&reply[s..=e])?),
        _ => anyhow::bail!("no JSON object in the reviewer's reply"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "\
diff --git a/m/pkg/widget/widget.go b/m/pkg/widget/widget.go
index 1111111..2222222 100644
--- a/m/pkg/widget/widget.go
+++ b/m/pkg/widget/widget.go
@@ -3,6 +3,7 @@ import (
 
 const (
 \tdefaultWidth = 10
+\tmaxWidgetWidth = 64
 )
 
@@ -20,6 +21,21 @@ type Widget struct {
 \tname string
+\tColor  string
+\tweight int
 }
 
+// ErrTooWide means the widget exceeds maxWidgetWidth.
+var ErrTooWide = errors.New(\"widget: too wide\")
+
+// Resize changes the width of the widget, refusing anything too wide.
+func (w *Widget) Resize(width int) error {
+\tif width > maxWidgetWidth {
+\t\treturn fmt.Errorf(\"widget %s: width %d exceeds the configured maximum\", w.name, width)
+\t}
+\tw.width = clampWidgetWidth(width, defaultWidth, maxWidgetWidth)
+\treturn nil
+}
+
+func clampWidgetWidth(v, lo, hi int) int { return min(max(v, lo), hi) }
+
 func unrelated() {}
";

    const TESTS: &str = "\
diff --git a/m/pkg/widget/widget_test.go b/m/pkg/widget/widget_test.go
index 3333333..4444444 100644
--- a/m/pkg/widget/widget_test.go
+++ b/m/pkg/widget/widget_test.go
@@ -1,3 +1,14 @@ package widget
 
+func TestResize(t *testing.T) {
+\tw := &Widget{name: \"a\", Color: \"red\"}
+\tif err := w.Resize(maxWidgetWidth + 1); err == nil {
+\t\tt.Fatal(\"expected an error\")
+\t}
+\tExpect(err).To(MatchError(\"widget a: width 65 exceeds the configured maximum\"))
+\tExpect(w.Resize(3)).To(Succeed())
+\tExpect(clampWidgetWidth(1, 2, 3)).To(Equal(2))
+\tmock.EXPECT().Store(gomock.Any()).Return(nil).Times(2)
+\tother.EXPECT().Load(gomock.Any()).Times(1)
+}
";

    fn dirs() -> Vec<String> {
        vec!["m/pkg/widget".to_string()]
    }

    #[test]
    fn declarations_come_from_added_lines_with_their_block_kind() {
        let got: Vec<(String, String, String)> = added_decls(SRC).into_iter()
            .map(|d| (d.name, d.kind, d.signature)).collect();
        let want = [
            ("maxWidgetWidth", "const", "maxWidgetWidth = 64"),
            ("Color", "field", "Color  string"),
            ("weight", "field", "weight int"),
            ("ErrTooWide", "var", "var ErrTooWide = errors.New(\"widget: too wide\")"),
            ("Resize", "method", "func (w *Widget) Resize(width int) error"),
            ("clampWidgetWidth", "func", "func clampWidgetWidth(v, lo, hi int) int"),
        ];
        let want: Vec<(String, String, String)> = want.iter()
            .map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string())).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn missing_symbols_are_read_from_the_parent_log() {
        let log = "# example.invalid/m/pkg/widget [example.invalid/m/pkg/widget.test]\n\
            pkg/widget/widget_test.go:4:12: w.Resize undefined (type *Widget has no field or method Resize)\n\
            pkg/widget/widget_test.go:4:21: undefined: maxWidgetWidth\n\
            pkg/widget/x_test.go:9:2: undefined: widget.ErrTooWide\n\
            pkg/widget/x_test.go:9:9: unknown field Color in struct literal of type Widget\n\
            pkg/widget/y_test.go:3:2: cannot find module providing package example.invalid/m/pkg/gizmo: module lookup disabled\n\
            pkg/widget/y_test.go:3:2: cannot find module providing package example.invalid/other: module lookup disabled\n";
        let got = log_symbols(log, Some("example.invalid/m"));
        let names: Vec<(&str, &str)> = got.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        assert_eq!(names, [("Resize", "field or method"), ("maxWidgetWidth", "identifier"),
                           ("ErrTooWide", "identifier"), ("Color", "field"), ("pkg/gizmo", "package")]);
        assert!(log_symbols(log, None).iter().all(|(_, k)| k != "package"));
    }

    #[test]
    fn required_api_joins_the_log_and_test_references() {
        let log = "a_test.go:1:1: undefined: widget.Gone\na_test.go:2:1: w.Resize undefined \
                   (type *Widget has no field or method Resize)\n";
        let req = required_api(log, None, SRC, TESTS, &dirs());
        let sig = |s: &str| req.iter().find(|r| r.symbol == s).map(|r| r.signature.as_str());
        assert_eq!(sig("Gone"), Some(""), "a log-only symbol keeps an empty signature");
        assert_eq!(sig("Resize"), Some("func (w *Widget) Resize(width int) error"));
        assert_eq!(sig("maxWidgetWidth"), Some("maxWidgetWidth = 64"));
        assert_eq!(sig("clampWidgetWidth"), Some("func clampWidgetWidth(v, lo, hi int) int"));
        assert_eq!(sig("Color"), Some("Color  string"));
        assert_eq!(sig("ErrTooWide"), None, "unreferenced declarations are not required");
        assert_eq!(sig("weight"), None);
        assert!(required_api(log, None, SRC, TESTS, &["elsewhere".into()]).iter()
            .all(|r| r.symbol == "Gone" || r.symbol == "Resize"));
    }

    #[test]
    fn exported_api_of_a_whole_file() {
        let src = "package widget\n\ntype Widget struct {\n\tName string\n\tsize int\n}\n\n\
                   func New() *Widget { return &Widget{} }\n\nfunc (w *Widget) size() int {\n\treturn 1\n}\n\n\
                   const (\n\tLimit = 3\n\tlow = 1\n)\n";
        let got: Vec<String> = exported_api(src, "w.go").into_iter().map(|d| d.signature).collect();
        assert_eq!(got, ["type Widget struct", "Name string", "func New() *Widget", "Limit = 3"]);
    }

    #[test]
    fn a_leaking_instruction_is_caught_and_a_clean_one_passes() {
        let sigs = vec!["func (w *Widget) Resize(width int) error".to_string(), "maxWidgetWidth = 64".to_string()];
        let clean = "## Widgets accept widths that are too large\n\nAdd `func (w *Widget) Resize(width int) error`. \
                     It must refuse a width above `maxWidgetWidth = 64` with an error naming the widget, \
                     and otherwise clamp the width into range.";
        let r = leak_check(clean, SRC, &sigs);
        assert!(!r.leaks(), "{r:?}");
        let verbatim = format!("{clean}\n\n```go\nw.width = clampWidgetWidth(width, defaultWidth, maxWidgetWidth)\n```");
        let r = leak_check(&verbatim, SRC, &sigs);
        assert!(r.leaks() && r.verbatim.len() == 1, "{r:?}");
        // Reformatted code is caught by the token shingles even without a
        // verbatim line.
        let reflowed = format!("{clean} Do: if width>maxWidgetWidth{{return fmt.Errorf(\"widget %s: width %d \
            exceeds the configured maximum\",w.name,width)}} w.width = clampWidgetWidth( width, defaultWidth, \
            maxWidgetWidth ) return nil");
        let r = leak_check(&reflowed, SRC, &sigs);
        assert!(r.verbatim.is_empty() && r.shingle_overlap > MAX_SHINGLE_OVERLAP, "{r:?}");
    }

    #[test]
    fn forbidden_mentions_are_listed() {
        assert_eq!(forbidden_mentions("Make the hidden tests pass"), ["hidden test"]);
        assert!(forbidden_mentions("Widgets refuse wide widths.").is_empty());
    }

    #[test]
    fn specificity_finds_each_kind_and_coverage_clears_them() {
        let f = specificity(TESTS, SRC, "", "", &dirs());
        let kinds: Vec<(&str, &str, bool)> = f.iter().map(|x| (x.kind.as_str(), x.detail.as_str(), x.covered)).collect();
        assert_eq!(kinds, [
            ("message", "widget a: width 65 exceeds the configured maximum", false),
            ("call_count", "Store.Times(2)", false),
            ("unexported", "func clampWidgetWidth", false),
            ("unexported", "const maxWidgetWidth", false),
        ]);
        let instr = "Resize must fail with `widget a: width 65 exceeds the configured maximum`; \
                     it calls Store twice; see maxWidgetWidth and clampWidgetWidth.";
        assert!(specificity(TESTS, SRC, instr, "", &dirs()).iter().all(|x| x.covered));
    }

    #[test]
    fn existing_and_echoed_strings_are_not_pinned() {
        let tests = "diff --git a/p/a_test.go b/p/a_test.go\n@@ -1,1 +1,3 @@\n \n\
            +\tin := \"a sufficiently long input string\"\n\
            +\tExpect(Echo(in)).To(Equal(\"a sufficiently long input string\"))\n\
            +\tExpect(err).To(MatchError(\"an error the parent already had\"))\n";
        assert!(specificity(tests, "", "", "return errors.New(\"an error the parent already had\")", &[]).is_empty());
        let clean = "diff --git a/p/a_test.go b/p/a_test.go\n@@ -1,1 +1,2 @@\n \n\
            +\tExpect(New().Len()).To(Equal(3))\n";
        assert!(specificity(clean, "", "", "", &[]).is_empty());
    }

    #[test]
    fn probe_replies_parse_strictly() {
        let p = parse_probe("Sure:\n```json\n{\"sufficient\": true, \"ambiguities\": [], \"guessed_names\": []}\n```").unwrap();
        assert!(p.sufficient && p.ambiguities.is_empty());
        assert!(parse_probe("{\"sufficient\": true}").is_err(), "all fields are required");
        assert!(parse_probe("no json").is_err());
    }

    #[test]
    fn a_patch_is_cut_down_to_the_tested_packages() {
        let two = format!("{TESTS}diff --git a/m/pkg/other/o_test.go b/m/pkg/other/o_test.go\n@@ -1 +1,2 @@\n \n+\tx()\n");
        assert_eq!(patch_in_dirs(&two, &dirs()), TESTS);
        assert_eq!(patch_in_dirs(&two, &[]), "");
    }

    #[test]
    fn words_and_bodies() {
        assert!(has_word("call Resize(3)", "Resize") && !has_word("ResizeAll", "Resize"));
        assert_eq!(strip_body("func F() int { return 1 }"), "func F() int");
        assert_eq!(strip_body("func G(f func() error) (x struct{}) {"), "func G(f func() error) (x struct{})");
        assert_eq!(strip_body("type T struct {"), "type T struct");
    }
}
