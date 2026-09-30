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

//! Which declared CSF components a task's reference fix touches.
//!
//! Each changed source file maps to the component whose declared `source`
//! is the longest path prefix of it (a component's source is a file or a
//! directory, relative to its model's root). A file no component declares
//! maps to none. Only components with a `source` can own files.

use super::model::LocatedModel;
use super::under;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The optional `csf` field of task.json.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskCsf {
    /// Architecture name(s) of the model(s) used, e.g. `["csf"]`.
    pub models: Vec<String>,
    /// Revision the model was read at, when known (not necessarily the
    /// task's commit: see README "CSF").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_rev: Option<String>,
    /// Components owning at least one changed source file, sorted. With more
    /// than one model a name is qualified as `architecture:component`.
    pub components: Vec<String>,
    /// Their kinds (`service`, `manager`, `library`, `adapter`, `gateway`,
    /// `resource`), sorted and deduplicated.
    pub kinds: Vec<String>,
    /// `existing`, `planned`, `mixed`, or absent when no component matched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Changed source files owned by some component, of `files_total`.
    pub files_mapped: usize,
    pub files_total: usize,
}

/// Map a task's changed source files onto the components of `models`.
pub fn map_files(models: &[LocatedModel], files: &[String], model_rev: Option<&str>) -> TaskCsf {
    let qualify = models.len() > 1;
    let mut components = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    let mut states = BTreeSet::new();
    let mut mapped = 0;
    for f in files {
        let best = models.iter().flat_map(|m| m.model.components.iter().map(move |c| (m, c)))
            .filter_map(|(m, c)| {
                let src = super::join(&m.root, c.source.as_deref()?);
                under(f, &src).then_some((src.len(), m, c))
            })
            .max_by_key(|(len, _, _)| *len);
        if let Some((_, m, c)) = best {
            mapped += 1;
            components.insert(if qualify { format!("{}:{}", m.model.architecture.name, c.name) } else { c.name.clone() });
            kinds.insert(c.kind.clone());
            states.insert(c.state.clone());
        }
    }
    let state = match states.len() {
        0 => None,
        1 => states.into_iter().next(),
        _ => Some("mixed".to_string()),
    };
    TaskCsf {
        models: models.iter().map(|m| m.model.architecture.name.clone()).collect(),
        model_rev: model_rev.map(str::to_string),
        components: components.into_iter().collect(),
        kinds: kinds.into_iter().collect(),
        state,
        files_mapped: mapped,
        files_total: files.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csf::model::tests::shop;

    fn map(files: &[&str]) -> TaskCsf {
        let files: Vec<String> = files.iter().map(|s| s.to_string()).collect();
        map_files(&[shop("svc")], &files, Some("abc"))
    }

    #[test]
    fn the_longest_declared_prefix_wins() {
        // services/orders owns the directory; services/orders/mail and the
        // single file services/orders/store/sql.go are longer declarations.
        assert_eq!(map(&["svc/services/orders/place.go"]).components, ["orders"]);
        assert_eq!(map(&["svc/services/orders/mail/send.go"]).components, ["mail"]);
        assert_eq!(map(&["svc/services/orders/store/sql.go"]).components, ["sql"]);
        assert_eq!(map(&["svc/services/orders/store/other.go"]).components, ["orders"],
                   "a file-level source owns only that file");
    }

    #[test]
    fn several_components_and_their_kinds_and_state() {
        let t = map(&["svc/services/orders/place.go", "svc/pkg/money/add.go", "svc/cmd/shop/main.go"]);
        assert_eq!(t.components, ["money", "orders", "wiring"]);
        assert_eq!(t.kinds, ["library", "manager", "service"]);
        assert_eq!(t.state.as_deref(), Some("existing"));
        assert_eq!((t.files_mapped, t.files_total), (3, 3));
        assert_eq!(t.models, ["shop"]);
        assert_eq!(t.model_rev.as_deref(), Some("abc"));
    }

    #[test]
    fn unowned_files_map_to_nothing() {
        let t = map(&["svc/tools/x.go", "services/orders/place.go", "svc/pkg/moneybags/x.go"]);
        assert!(t.components.is_empty() && t.kinds.is_empty());
        assert_eq!(t.state, None);
        assert_eq!((t.files_mapped, t.files_total), (0, 3));
        assert_eq!(map_files(&[], &["a.go".to_string()], None).files_mapped, 0);
    }

    #[test]
    fn planned_and_existing_together_are_mixed_and_two_models_qualify_names() {
        let mut planned = shop("svc");
        planned.model.components[2].state = "planned".into();
        let files = vec!["svc/pkg/money/a.go".to_string(), "svc/services/orders/b.go".to_string()];
        assert_eq!(map_files(&[planned], &files, None).state.as_deref(), Some("mixed"));
        let mut other = shop("other");
        other.model.architecture.name = "depot".into();
        let files = vec!["svc/pkg/money/a.go".to_string(), "other/pkg/money/a.go".to_string()];
        assert_eq!(map_files(&[shop("svc"), other], &files, None).components, ["depot:money", "shop:money"]);
    }
}
