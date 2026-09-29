//! Process exit codes of design §2.4, matching `worker/__main__.py`.

use std::process::ExitCode;

/// Every way the process ends, each with one fixed exit code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exit {
    /// 0: clean shutdown or a config restart directive.
    CleanShutdown,
    /// 1: runtime error.
    Runtime,
    /// 2: config or CLI error.
    Config,
    /// 3: refuse to start (GPU lease or identity mismatch).
    RefuseToStart,
    /// 4: fatal accelerator fault.
    FatalAccelerator,
}

impl Exit {
    pub const fn code(self) -> u8 {
        match self {
            Self::CleanShutdown => 0,
            Self::Runtime => 1,
            Self::Config => 2,
            Self::RefuseToStart => 3,
            Self::FatalAccelerator => 4,
        }
    }
}

impl From<Exit> for ExitCode {
    fn from(exit: Exit) -> Self {
        Self::from(exit.code())
    }
}
