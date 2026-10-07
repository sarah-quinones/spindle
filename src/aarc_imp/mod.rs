use crate::mock::atomic::Ordering::*;
use crate::mock::atomic::{AtomicPtr, AtomicUsize};
use crate::mock::{Arc, spin_loop};
use std::ops::Deref;
use std::ptr::null_mut;

/// Keeps a successful load alive while `clear` owns final reclamation.
///
/// An installed value owns one counter slot; each in-flight load reserves another.
/// Dropping a guard releases its load reservation. After removing the raw pointer,
/// `clear` waits for all load reservations to end, then releases the stored `Arc`.
struct Guard<'a, T> {
	ptr: *const T,
	count: &'a AtomicUsize,
}

impl<T> Deref for Guard<'_, T> {
	type Target = T;

	fn deref(&self) -> &Self::Target {
		unsafe { &*self.ptr }
	}
}

impl<T> Drop for Guard<'_, T> {
	fn drop(&mut self) {
		let old = self.count.fetch_sub(1, Release);
		debug_assert!(old > 1);
	}
}

#[repr(C)]
pub struct AtomicArc<T> {
	raw: AtomicPtr<T>,
	/// After initialization, every change must be an atomic read-modify-write.
	/// This preserves the release sequences from publication and guard drops that
	/// synchronize the relaxed raw-pointer accesses. Other modules may only load it.
	pub(crate) count: AtomicUsize,
}

impl<T> AtomicArc<T> {
	#[inline(always)]
	pub fn null() -> Self {
		Self {
			raw: AtomicPtr::new(null_mut()),
			count: AtomicUsize::new(0),
		}
	}

	#[inline]
	pub fn is_null(this: &Self) -> bool {
		this.raw.load(Relaxed).is_null()
	}

	#[inline]
	pub fn is_free(this: &Self) -> bool {
		this.raw.load(Relaxed).is_null() && this.count.load(Relaxed) == 0
	}

	#[inline(never)]
	pub fn load(this: &Self) -> Option<impl '_ + Deref<Target = T>> {
		if !Self::reserve_load(this) {
			return None;
		}
		Self::finish_load(this)
	}

	/// Reserves a load slot unless the stored reference is already absent.
	#[inline(always)]
	fn reserve_load(this: &Self) -> bool {
		let old = this.count.fetch_add(1, Acquire);
		if old == 0 {
			Self::release_load(this);
			false
		} else {
			true
		}
	}

	/// Finishes a reserved load and transfers its slot to the returned guard.
	#[inline(always)]
	fn finish_load(this: &Self) -> Option<Guard<'_, T>> {
		let raw = this.raw.load(Relaxed);
		if raw.is_null() {
			Self::release_load(this);
			return None;
		}

		Some(Guard {
			ptr: raw,
			count: &this.count,
		})
	}

	/// Releases a load slot that did not become a guard.
	#[inline(always)]
	fn release_load(this: &Self) {
		let old = this.count.fetch_sub(1, Relaxed);
		debug_assert!(old > 0);
	}

	/// Removes and reclaims the stored reference after active loads finish.
	///
	/// The `compare_exchange_null` that publishes the current generation must
	/// finish before this operation starts. A new generation may be published
	/// after reclamation releases the count to zero, even while the removed
	/// `Arc` is being dropped.
	/// Callers must let every existing guard finish independently of this call.
	/// Spindle satisfies that invariant by clearing child schedulers only after
	/// their completion barrier; each child slot also has exactly one clearer.
	#[inline(never)]
	pub fn clear(this: &Self) {
		let ptr = this.raw.load(Relaxed);
		this.raw.store(null_mut(), Relaxed);
		Self::reclaim(this, ptr);
	}

	/// Reclaims a removed stored reference after every active load leaves.
	#[inline(always)]
	fn reclaim(this: &Self, ptr: *const T) {
		if !ptr.is_null() {
			loop {
				while this.count.load(Relaxed) != 1 {
					spin_loop();
				}
				if this.count.compare_exchange_weak(1, 0, AcqRel, Relaxed).is_ok() {
					break;
				}
			}

			unsafe { Arc::decrement_strong_count(ptr) };
		}
	}

	#[inline(never)]
	pub fn compare_exchange_null(this: &Self, new: Arc<T>) -> Result<*const T, Arc<T>> {
		if this.count.compare_exchange(0, 1, AcqRel, Relaxed).is_err() {
			return Err(new);
		}

		let ptr = Arc::into_raw(new) as *mut T;
		this.raw.store(ptr, Relaxed);

		Ok(ptr)
	}
}

#[cfg(all(test, loom))]
mod tests {
	use super::AtomicArc;
	use crate::mock::Arc;
	use crate::mock::atomic::Ordering::SeqCst;
	use crate::mock::atomic::{AtomicBool, AtomicUsize};

	/// Records whether the stored `Arc` was reclaimed and how often.
	struct DropProbe {
		drops: Arc<AtomicUsize>,
		value: loom::cell::Cell<usize>,
	}

	impl DropProbe {
		/// Tracks the payload write before publication; readers must acquire its visibility.
		fn new(drops: Arc<AtomicUsize>, payload: usize) -> Self {
			let value = loom::cell::Cell::new(0);
			value.set(payload);
			Self { drops, value }
		}
	}

	impl Drop for DropProbe {
		fn drop(&mut self) {
			// Track reclamation before the counter can synchronize with readers.
			self.value.set(0);
			self.drops.fetch_add(1, SeqCst);
		}
	}

	fn publish(slot: &AtomicArc<DropProbe>, drops: &Arc<AtomicUsize>, payload: usize) {
		let value = Arc::new(DropProbe::new(drops.clone(), payload));
		assert!(AtomicArc::compare_exchange_null(slot, value).is_ok());
	}

	/// Checks that clear cannot return or reclaim while load guards are alive.
	#[test]
	fn clear_waits_for_multiple_loads() {
		loom::model(|| {
			let drops = Arc::new(AtomicUsize::new(0));
			let slot = Arc::new(AtomicArc::null());
			publish(&slot, &drops, 7);

			let first = AtomicArc::load(&slot).unwrap();
			let second = AtomicArc::load(&slot).unwrap();
			assert_eq!(first.value.get(), 7);
			assert_eq!(second.value.get(), 7);
			assert_eq!(slot.count.load(SeqCst), 3);

			let clear_slot = slot.clone();
			let clear_done = Arc::new(AtomicBool::new(false));
			let clear_done_thread = clear_done.clone();
			let clear = loom::thread::spawn(move || {
				AtomicArc::clear(&clear_slot);
				clear_done_thread.store(true, SeqCst);
			});
			// Observe pointer removal while both load reservations remain alive.
			while !AtomicArc::is_null(&slot) {
				loom::thread::yield_now();
			}
			assert_eq!(drops.load(SeqCst), 0);
			assert!(!clear_done.load(SeqCst));

			drop(first);
			drop(second);
			clear.join().unwrap();

			assert!(clear_done.load(SeqCst));
			assert_eq!(drops.load(SeqCst), 1);
			assert!(AtomicArc::is_free(&slot));
		});
	}

	/// Checks unfenced reads and republication against the previous generation's clear.
	#[test]
	fn concurrent_clear_and_republish_keep_generations_separate() {
		let mut builder = loom::model::Builder::new();
		builder.max_branches = 2000;
		builder.check(|| {
			let old_drops = Arc::new(AtomicUsize::new(0));
			let new_drops = Arc::new(AtomicUsize::new(0));
			let slot = Arc::new(AtomicArc::null());
			publish(&slot, &old_drops, 1);

			let clear_slot = slot.clone();
			let clear = loom::thread::spawn(move || AtomicArc::clear(&clear_slot));

			let publish_slot = slot.clone();
			let publish_drops = new_drops.clone();
			let republish = loom::thread::spawn(move || {
				let value = Arc::new(DropProbe::new(publish_drops, 2));
				AtomicArc::compare_exchange_null(&publish_slot, value).is_ok()
			});

			// No channel or join orders the clear before the publisher or this
			// late load. A guard must keep whichever generation it sees alive.
			if let Some(late) = AtomicArc::load(&slot) {
				match late.value.get() {
					1 => assert_eq!(old_drops.load(SeqCst), 0),
					2 => assert_eq!(new_drops.load(SeqCst), 0),
					_ => unreachable!(),
				}
			}

			let republished = republish.join().unwrap();
			clear.join().unwrap();
			assert_eq!(old_drops.load(SeqCst), 1);

			// Publication may legitimately lose to an active old generation or
			// an empty-slot load reservation. Successful publication must survive
			// the old clearer without relying on that thread having returned.
			if republished {
				assert_eq!(AtomicArc::load(&slot).unwrap().value.get(), 2);
				AtomicArc::clear(&slot);
			}
			assert_eq!(new_drops.load(SeqCst), 1);
			assert!(AtomicArc::is_free(&slot));
		});
	}

	/// Checks that old-destructor completion cannot invalidate a new generation.
	#[test]
	fn republish_while_previous_value_is_dropping() {
		/// Holds the old destructor open after its stored-reference slot is released.
		struct BlockingDrop {
			probe: DropProbe,
			gate: Option<(loom::sync::mpsc::Sender<()>, loom::sync::Mutex<loom::sync::mpsc::Receiver<()>>)>,
		}

		impl Drop for BlockingDrop {
			fn drop(&mut self) {
				if let Some((entered, resume)) = &self.gate {
					entered.send(()).unwrap();
					resume.lock().unwrap().recv().unwrap();
				}
			}
		}

		loom::model(|| {
			let old_drops = Arc::new(AtomicUsize::new(0));
			let new_drops = Arc::new(AtomicUsize::new(0));
			let slot = Arc::new(AtomicArc::null());
			let (entered_tx, entered_rx) = loom::sync::mpsc::channel();
			let (resume_tx, resume_rx) = loom::sync::mpsc::channel();
			assert!(
				AtomicArc::compare_exchange_null(
					&slot,
					Arc::new(BlockingDrop {
						probe: DropProbe::new(old_drops.clone(), 1),
						gate: Some((entered_tx, loom::sync::Mutex::new(resume_rx))),
					}),
				)
				.is_ok()
			);

			let clear_slot = slot.clone();
			let clear = loom::thread::spawn(move || AtomicArc::clear(&clear_slot));
			entered_rx.recv().unwrap();
			assert_eq!(old_drops.load(SeqCst), 0);
			assert!(AtomicArc::is_free(&slot));
			assert!(
				AtomicArc::compare_exchange_null(
					&slot,
					Arc::new(BlockingDrop {
						probe: DropProbe::new(new_drops.clone(), 2),
						gate: None,
					}),
				)
				.is_ok()
			);
			let loaded = AtomicArc::load(&slot).unwrap();
			assert_eq!(loaded.probe.value.get(), 2);

			// Retain a new-generation guard across the old destructor's completion.
			resume_tx.send(()).unwrap();
			clear.join().unwrap();
			assert_eq!(old_drops.load(SeqCst), 1);
			assert_eq!(new_drops.load(SeqCst), 0);
			assert_eq!(loaded.probe.value.get(), 2);
			drop(loaded);
			assert_eq!(AtomicArc::load(&slot).unwrap().probe.value.get(), 2);

			AtomicArc::clear(&slot);
			assert_eq!(old_drops.load(SeqCst), 1);
			assert_eq!(new_drops.load(SeqCst), 1);
			assert!(AtomicArc::is_free(&slot));
		});
	}

	/// Checks publication and payload visibility while two empty-slot loads overlap.
	#[test]
	fn empty_reservations_preserve_publication() {
		// This flag records coverage across models and never synchronizes their
		// actors. Both reservations must be observed before publication is tried.
		let saw_two = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
		let observed = saw_two.clone();
		let mut builder = loom::model::Builder::new();
		builder.max_branches = 2000;
		// Limited or checkpointed exploration may omit the overlap. Safety
		// assertions still apply to every execution; require coverage only when
		// exploring all interleavings from the beginning.
		let check_coverage = builder.preemption_bound.is_none()
			&& builder.max_permutations.is_none()
			&& builder.max_duration.is_none()
			&& builder.checkpoint_file.is_none();
		builder.check(move || {
			let drops = Arc::new(AtomicUsize::new(0));
			let slot = Arc::new(AtomicArc::<DropProbe>::null());
			let readers = (0..2)
				.map(|_| {
					let slot = slot.clone();
					loom::thread::spawn(move || {
						if let Some(value) = AtomicArc::load(&slot) {
							assert_eq!(value.value.get(), 7);
						}
					})
				})
				.collect::<Vec<_>>();

			if slot.count.load(SeqCst) == 2 {
				observed.store(true, SeqCst);
			}
			let value = Arc::new(DropProbe::new(drops.clone(), 7));
			let published = AtomicArc::compare_exchange_null(&slot, value).is_ok();
			for reader in readers {
				reader.join().unwrap();
			}
			if published {
				assert_eq!(AtomicArc::load(&slot).unwrap().value.get(), 7);
				AtomicArc::clear(&slot);
			}
			assert_eq!(drops.load(SeqCst), 1);
			assert!(AtomicArc::is_free(&slot));
		});
		assert!(
			!check_coverage || saw_two.load(SeqCst),
			"model did not exercise two empty-slot reservations"
		);
	}
}

#[cfg(all(test, not(loom), not(miri)))]
mod regression_tests {
	use super::AtomicArc;
	use crate::mock::Arc;
	use crate::mock::atomic::AtomicUsize;
	use crate::mock::atomic::Ordering::SeqCst;
	use std::time::Duration;

	/// Counts reclamation of the stored `Arc`, without an unrelated payload.
	struct DropCounter(Arc<AtomicUsize>);

	impl Drop for DropCounter {
		fn drop(&mut self) {
			self.0.fetch_add(1, SeqCst);
		}
	}

	/// Forces a reserved load to observe null and checks that clear still reclaims its Arc.
	/// The null observation does not order the legacy counter decrements;
	/// separate Loom models cover reclamation and load-reservation release ordering.
	#[test]
	fn clear_reclaims_null_after_increment() {
		let drops = Arc::new(AtomicUsize::new(0));
		let slot = Arc::new(AtomicArc::null());
		assert!(AtomicArc::compare_exchange_null(&slot, Arc::new(DropCounter(drops.clone()))).is_ok());

		let (incremented_tx, incremented_rx) = std::sync::mpsc::channel();
		let load_slot = slot.clone();
		let load = std::thread::spawn(move || {
			// Split the load at its reservation and release it if synchronization panics.
			assert!(AtomicArc::reserve_load(&load_slot));
			let wait = std::panic::catch_unwind(|| {
				incremented_tx.send(()).unwrap();
				let started = std::time::Instant::now();
				while !AtomicArc::is_null(&load_slot) {
					assert!(started.elapsed() < Duration::from_secs(30), "clear did not reach null");
					std::thread::yield_now();
				}
			});
			if let Err(panic) = wait {
				AtomicArc::release_load(&load_slot);
				std::panic::resume_unwind(panic);
			}
			assert!(AtomicArc::finish_load(&load_slot).is_none());
		});

		incremented_rx.recv_timeout(Duration::from_secs(30)).unwrap();
		let (clear_done_tx, clear_done_rx) = std::sync::mpsc::channel();
		let clear_slot = slot.clone();
		let clear = std::thread::spawn(move || {
			AtomicArc::clear(&clear_slot);
			clear_done_tx.send(()).unwrap();
		});
		clear_done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
		load.join().unwrap();
		clear.join().unwrap();

		assert_eq!(drops.load(SeqCst), 1);
		assert!(AtomicArc::is_free(&slot));
	}
}
