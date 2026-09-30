//! Policy thread glue (design §2.3): pose packets into the `seeon-worker`
//! domain, fall and bed accelerator requests, observation coverage (G16) and
//! durable event identity (G20).

pub mod bed;
pub mod coverage;
pub mod fall;
pub mod identity;
pub mod ingest;
