//! Kernel-lock oracle for root's unclosed-media safety path, not private-field inspection.

use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use super::models::ModelOwners;
use crate::gpu::lease::{LeaseError, acquire};
use crate::shutdown::ShutdownDeadline;

#[test]
fn retained_media_lease_survives_gpu_owner_drop() -> Result<(), Box<dyn Error>> {
    let root = std::env::temp_dir().join(format!("seeon-s4-retained-lease-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let lease = acquire(&root).map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut owner = ModelOwners::new(
        lease,
        Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?),
    );
    owner.retain_lease_until_process_exit();
    owner.retain_lease_until_process_exit();
    drop(owner);
    let refused = matches!(acquire(&root), Err(LeaseError::Unavailable { .. }));
    std::fs::remove_dir_all(&root)?;
    assert!(refused);
    Ok(())
}

#[test]
fn closed_owner_drop_releases_the_kernel_lease() -> Result<(), Box<dyn Error>> {
    let root = std::env::temp_dir().join(format!("seeon-s4-closed-lease-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let lease = acquire(&root).map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let owner = ModelOwners::new(
        lease,
        Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?),
    );
    drop(owner);
    let reacquired = acquire(&root);
    let admitted = reacquired.is_ok();
    drop(reacquired);
    std::fs::remove_dir_all(&root)?;
    assert!(admitted);
    Ok(())
}
