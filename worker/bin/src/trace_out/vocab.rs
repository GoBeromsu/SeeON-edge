//! The closed string vocabularies of `contracts/replay_trace.py`.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lifecycle {
    New,
    Tracked,
    Shadow,
    Lost,
}

impl Lifecycle {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Tracked => "tracked",
            Self::Shadow => "shadow",
            Self::Lost => "lost",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceEvent {
    Open,
    Frame,
    Reconnect,
    Lost,
}

impl SourceEvent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Frame => "frame",
            Self::Reconnect => "reconnect",
            Self::Lost => "lost",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    LegacyAssociation,
    Nvdcf,
}

impl Source {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyAssociation => "legacy-association",
            Self::Nvdcf => "nvdcf",
        }
    }
}
