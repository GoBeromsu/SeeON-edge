//! Identity placeholders (`_verified_identity_field`,
//! `worker/runtime/model_composition.py` L184-223): a component's compiled
//! identity against the identity the runtime resolved. A compiled
//! `RuntimeResolvedIdentityField` marker (or `None`) accepts any non-empty
//! resolved string; a compiled string must be concrete and equal it.

/// A text that stands in for an identity instead of naming one.
const PLACEHOLDER: &str = "runtime-resolved";

/// A compiled identity field as `ComponentBinding` carries it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Compiled<'a> {
    /// `None`.
    Absent,
    /// A `RuntimeResolvedIdentityField` marker such as
    /// `RUNTIME_RESOLVED_ARTIFACT_DIGEST`.
    RuntimeResolved,
    /// A `str`.
    Text(&'a str),
    /// Any other value, such as an `int`.
    NotText,
}

/// The binding the component was compiled from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binding<'a> {
    pub component_id: &'a str,
    pub compiled: Compiled<'a>,
}

/// One component attribute as Python's `getattr(component, name, None)`
/// returns it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Attr<'a> {
    /// Missing, or `None`: the chain moves to the next candidate.
    #[default]
    Absent,
    /// A `str`.
    Text(&'a str),
    /// Any other present value, such as an `int`. It is not `None`, so the
    /// chain stops here and the `str` check refuses it.
    NotText,
}

/// The component attributes Python reads: `<field>`, then `_<field>`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Attrs<'a> {
    pub public: Attr<'a>,
    pub private: Attr<'a>,
}

/// What the `getattr` chain and `provisioned` leave for the `str` check.
fn chain<'a>(attrs: Attrs<'a>, provisioned: Option<&'a str>) -> Option<&'a str> {
    match (attrs.public, attrs.private) {
        (Attr::Text(text), _) | (Attr::Absent, Attr::Text(text)) => Some(text),
        (Attr::NotText, _) | (Attr::Absent, Attr::NotText) => None,
        (Attr::Absent, Attr::Absent) => provisioned,
    }
}

/// Which `DetectionModuleActivationError` message refused the component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaceholderKind {
    /// `component {id!r} has no compiled {label} identity`.
    NoCompiled,
    /// `component {id!r} has no resolved {label} identity`.
    NoResolved,
    /// `component {id!r} {label} identity mismatch: compiled {expected!r},
    /// resolved {resolved!r}`.
    Mismatch,
}

/// A refusal and what the Python message names. The subject is
/// `"{expected} {resolved}"` for `Mismatch` and empty otherwise.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlaceholderError {
    pub kind: PlaceholderKind,
    pub component: String,
    pub label: String,
    pub subject: String,
}

/// `_verified_identity_field(binding, component, field, provisioned, label,
/// expected=...)`: the resolved identity, or the refusal. `expected` of
/// `None` or `Some(Compiled::Absent)` uses the binding's compiled value.
pub fn verified_identity_field(
    binding: &Binding<'_>,
    attrs: Attrs<'_>,
    provisioned: Option<&str>,
    label: &str,
    expected: Option<Compiled<'_>>,
) -> Result<String, PlaceholderError> {
    let refuse = |kind, subject: String| PlaceholderError {
        kind,
        component: binding.component_id.to_owned(),
        label: label.to_owned(),
        subject,
    };
    let expected = match expected {
        None | Some(Compiled::Absent) => binding.compiled,
        Some(value) => value,
    };
    let resolved = chain(attrs, provisioned);
    match expected {
        Compiled::Absent | Compiled::RuntimeResolved => match resolved {
            Some(text) if !text.is_empty() => Ok(text.to_owned()),
            _ => Err(refuse(PlaceholderKind::NoResolved, String::new())),
        },
        Compiled::Text(compiled) if !compiled.is_empty() && !compiled.contains(PLACEHOLDER) => {
            match resolved {
                Some(text) if !text.is_empty() && !text.contains(PLACEHOLDER) => {
                    if text == compiled {
                        Ok(text.to_owned())
                    } else {
                        Err(refuse(
                            PlaceholderKind::Mismatch,
                            format!("{compiled} {text}"),
                        ))
                    }
                }
                _ => Err(refuse(PlaceholderKind::NoResolved, String::new())),
            }
        }
        Compiled::Text(_) | Compiled::NotText => {
            Err(refuse(PlaceholderKind::NoCompiled, String::new()))
        }
    }
}
