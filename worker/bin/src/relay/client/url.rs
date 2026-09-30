//! Python `normalize_http_base`, `join_http_url` and `urlencode` for the
//! relay client. Pure string work.

use ureq::http::Uri;

use super::ConfigError;

/// Python `normalize_http_base`, stricter: no query, fragment, whitespace or
/// control character. Returns the URL without trailing slashes.
pub(super) fn normalize_base(value: &str) -> Result<String, ConfigError> {
    if value
        .chars()
        .any(|c| c.is_whitespace() || c.is_control() || c == '?' || c == '#')
    {
        return Err(ConfigError::Base);
    }
    let scheme = ["http://", "https://"]
        .into_iter()
        .find(|scheme| {
            value
                .get(..scheme.len())
                .is_some_and(|head| head.eq_ignore_ascii_case(scheme))
        })
        .ok_or(ConfigError::Scheme)?;
    let netloc = value[scheme.len()..].split('/').next().unwrap_or_default();
    if netloc.is_empty() {
        return Err(ConfigError::Host);
    }
    if netloc.contains('@') {
        return Err(ConfigError::Credentials);
    }
    let base = value.trim_end_matches('/').to_owned();
    Uri::try_from(format!("{base}/")).map_err(|_| ConfigError::Base)?;
    Ok(base)
}

/// Python `join_http_url(base, path)`, then `?` and `urlencode(query)` when
/// the query is not empty.
pub(super) fn join(base: &str, path: &str, query: &[(&str, &str)]) -> String {
    let mut url = format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    for (index, (key, value)) in query.iter().enumerate() {
        url.push(if index == 0 { '?' } else { '&' });
        url.push_str(&quote_plus(key));
        url.push('=');
        url.push_str(&quote_plus(value));
    }
    url
}

/// Python `quote_plus`: unreserved bytes pass, space is `+`, every other
/// byte of the UTF-8 form is `%XX`.
fn quote_plus(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                out.push(char::from(byte));
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}
