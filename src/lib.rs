use Ordering::*;
use barrier::{Barrier, BarrierInit, ExitStatus};
use mock::atomic::*;
use mock::{Arc, Cell, UnsafeCell};
use rayon::iter::plumbing::{Producer, ProducerCallback};
use std::any::Any;
use std::iter;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::ptr::{null, null_mut};

#[cfg(loom)]
use loom::thread;
#[cfg(miri)]
use std::thread;

mod mock {
	#[cfg(not(loom))]
	pub use {
		core::cell::{Cell, UnsafeCell},
		core::hint::spin_loop,
		std::sync::*,
	};

	#[cfg(loom)]
	pub use {
		loom::cell::{Cell, UnsafeCell},
		loom::hint::spin_loop,
		loom::sync::*,
	};
}

#[cfg(loom)]
mod atomic_wait_imp {
	pub use loom::sync::atomic::AtomicU32;
	pub fn wake_all(_atomic: *const AtomicU32) {
		loom::thread::yield_now();
	}
	pub fn wait(_atomic: &AtomicU32, _value: u32) {
		loom::thread::yield_now();
	}
}

#[cfg(not(loom))]
mod atomic_wait_imp {
	pub use atomic_wait::{wait, wake_all};
}

use equator::assert;

use rayon::prelude::*;

mod aarc_imp;
use aarc_imp::AtomicArc;

#[cfg(not(any(loom, miri)))]
type ThdScope<'a, 'b> = &'a rayon::Scope<'b>;
#[cfg(any(loom, miri))]
type ThdScope<'a, 'b> = &'a &'b ();

mod defer {
	use super::*;
	pub struct Defer<F: FnOnce()>(ManuallyDrop<F>);
	impl<F: FnOnce()> Drop for Defer<F> {
		fn drop(&mut self) {
			unsafe { ManuallyDrop::take(&mut self.0)() }
		}
	}
	pub fn defer<F: FnOnce()>(f: F) -> defer::Defer<F> {
		defer::Defer(ManuallyDrop::new(f))
	}
}
use defer::defer;

#[derive(Debug)]
#[repr(transparent)]
struct SyncUnsafeCell<T: ?Sized>(pub UnsafeCell<T>);

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
struct SyncCell<T: ?Sized>(T);
unsafe impl<T> Sync for SyncCell<T> {}
unsafe impl<T> Send for SyncCell<T> {}

impl<T> SyncUnsafeCell<T> {
	#[inline(always)]
	pub fn new(inner: T) -> Self {
		Self(UnsafeCell::new(inner))
	}
}

#[cfg(not(loom))]
pub const DEFAULT_SPIN_LIMIT: u32 = 65536;
#[cfg(not(loom))]
pub const DEFAULT_PAUSE_LIMIT: u32 = 8;

#[cfg(loom)]
pub const DEFAULT_SPIN_LIMIT: u32 = 1;
#[cfg(loom)]
pub const DEFAULT_PAUSE_LIMIT: u32 = 1;

pub static SPIN_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(DEFAULT_SPIN_LIMIT);
pub static PAUSE_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(DEFAULT_PAUSE_LIMIT);

mod barrier;

type PanicLoad = Box<dyn Any + Send>;
type TaskInner = dyn Sync + Fn(usize, &AtomicPtr<PanicLoad>);

struct Worker {
	init: Arc<BarrierInit>,
	thread_id_used: Box<[AtomicU32]>,

	task: AtomicPtr<*const TaskInner>,
	panic_slot: AtomicPtr<PanicLoad>,
	n_jobs: AtomicUsize,
	structured_jobs: Box<[AtomicUsize]>,
	unstructured_jobs: AtomicUsize,
	leader: Barrier,
}

impl Worker {
	fn new(n_threads: usize) -> Self {
		let init = BarrierInit::new();
		let thread_id_used = iter::repeat_n(0, n_threads.div_ceil(32)).map(AtomicU32::new).collect::<Box<[_]>>();

		thread_id_used[0].store(1, Relaxed);
		Self {
			thread_id_used,
			task: AtomicPtr::new(null_mut()),
			panic_slot: AtomicPtr::new(null_mut()),
			n_jobs: AtomicUsize::new(0),
			structured_jobs: iter::repeat_n(0, n_threads).map(AtomicUsize::new).collect(),
			unstructured_jobs: AtomicUsize::new(0),
			leader: BarrierInit::barrier(&init, 0),
			init,
		}
	}
}

struct Womanager {
	n_threads: usize,
	waker: AtomicU32,
	spawn: &'static (dyn Sync + Fn(&Womanager)),
	scope: Option<ThdScope<'static, 'static>>,
	n_spawn: &'static AtomicUsize,

	worker: Worker,

	children: Box<[AtomicArc<(AtomicUsize, Womanager)>]>,
	children_thread_requests: AtomicUsize,

	init: Arc<BarrierInit>,
	announce_exit: AtomicBool,
	announce_rest: AtomicBool,
	depth: usize,
}

#[cfg(not(loom))]
thread_local! {
	static WORKER: Cell<bool> = const { Cell::new(true) };
	static WOMANAGER: Cell<*const Womanager> = const { Cell::new(null()) };
	static ROOT: Cell<*const Womanager> = const { Cell::new(null()) };
	static THREAD_ID: Cell<usize> = const { Cell::new(0) };
}
#[cfg(loom)]
loom::thread_local! {
	static WORKER: Cell<bool> = Cell::new(true) ;
	static WOMANAGER: Cell<*const Womanager> = Cell::new(null()) ;
	static ROOT: Cell<*const Womanager> = Cell::new(null()) ;
	static THREAD_ID: Cell<usize> = Cell::new(0) ;
}

pub fn current_num_threads() -> usize {
	let root = WOMANAGER.with(|x| x.get());
	if !root.is_null() {
		unsafe { &*root }.worker.init.current_num_threads()
	} else {
		rayon::current_num_threads()
	}
}

pub fn max_num_threads() -> usize {
	let root = WOMANAGER.with(|x| x.get());
	if !root.is_null() {
		unsafe { &*root }.n_threads
	} else {
		rayon::current_num_threads()
	}
}

pub fn current_thread_index() -> Option<usize> {
	let root = WOMANAGER.with(|x| x.get());
	if !root.is_null() {
		if WORKER.with(|x| x.get()) {
			Some(THREAD_ID.with(|x| x.get()))
		} else {
			None
		}
	} else {
		rayon::current_thread_index()
	}
}

pub fn with_lock<R: Send>(max_threads: usize, f: impl Send + FnOnce() -> R) -> R {
	assert!(max_threads != 0);
	#[cfg(miri)]
	let n_threads = 4;

	#[cfg(loom)]
	let n_threads = 2;

	#[cfg(not(any(loom, miri)))]
	let n_threads = rayon::current_num_threads();

	let n_threads = Ord::min(n_threads, max_threads);
	let root = ROOT.with(|x| x.get());

	let n_spawn = &AtomicUsize::new(0);
	let spawn: &(dyn Sync + Fn(&Womanager)) = &|womanager: &Womanager| worker_loop(womanager);
	let mut womanager = Womanager {
		n_threads,
		waker: AtomicU32::new(0),
		spawn: unsafe { core::mem::transmute(spawn) },
		scope: None,
		n_spawn: unsafe { &*(n_spawn as *const _) },
		worker: Worker::new(n_threads),
		children: iter::repeat_n((), n_threads).map(|()| AtomicArc::null()).collect(),
		children_thread_requests: AtomicUsize::new(0),
		init: BarrierInit::new(),
		announce_exit: AtomicBool::new(false),
		announce_rest: AtomicBool::new(true),
		depth: unsafe { WOMANAGER.with(|x| x.get()).as_ref().map(|w| w.depth + 1).unwrap_or(0) },
	};

	if root.is_null() {
		#[cfg(any(loom, miri))]
		{
			let mut threads = vec![];

			womanager.scope = Some(&&());
			let barrier = BarrierInit::barrier(&womanager.init, 0);
			womanager.n_spawn.store(n_threads, Relaxed);
			for _ in 1..n_threads {
				let spawn = womanager.spawn;
				let womanager = unsafe { &*&raw const womanager };
				threads.push(thread::spawn(move || spawn(womanager)));
			}

			let old_root = ROOT.with(|x| x.replace(&raw const womanager));
			let old_womanager = WOMANAGER.with(|x| x.replace(&raw const womanager));
			let old_thread_id = THREAD_ID.with(|x| x.replace(0));
			let old_worker = WORKER.with(|x| x.replace(false));

			let __guard__ = defer(|| {
				womanager.waker.store(1, Release);
				womanager.announce_exit.store(true, Release);
				womanager.announce_rest.store(false, Release);
				atomic_wait_imp::wake_all(&womanager.waker);
				ROOT.with(|x| x.set(old_root));
				THREAD_ID.with(|x| x.set(old_thread_id));
				WORKER.with(|x| x.set(old_worker));
				WOMANAGER.with(|x| x.set(old_womanager));
				barrier.wait();
				for thd in threads {
					thd.join().unwrap();
				}
			});

			f()
		}

		#[cfg(not(any(loom, miri)))]
		rayon::scope(|scope| {
			womanager.scope = Some(unsafe { &*(scope as *const _ as *const _) });

			let barrier = BarrierInit::barrier(&womanager.init, 0);
			womanager.n_spawn.store(n_threads, Relaxed);
			for _ in 1..n_threads {
				scope.spawn(|_| (womanager.spawn)(&womanager));
			}

			let old_root = ROOT.with(|x| x.replace(&raw const womanager));
			let old_womanager = WOMANAGER.with(|x| x.replace(&raw const womanager));
			let old_thread_id = THREAD_ID.with(|x| x.replace(0));
			let old_worker = WORKER.with(|x| x.replace(false));

			let __guard__ = defer(|| {
				womanager.waker.store(1, Release);
				womanager.announce_exit.store(true, Release);
				womanager.announce_rest.store(false, Release);
				atomic_wait_imp::wake_all(&womanager.waker);
				ROOT.with(|x| x.set(old_root));
				THREAD_ID.with(|x| x.set(old_thread_id));
				WORKER.with(|x| x.set(old_worker));
				WOMANAGER.with(|x| x.set(old_womanager));
				barrier.wait();
			});

			f()
		})
	} else {
		if !WORKER.with(|x| x.get()) {
			return f();
		}
		let barrier = BarrierInit::barrier(&womanager.init, 0);

		let mut womanager = Some(Arc::new((AtomicUsize::new(n_threads - 1), womanager)));
		let root = unsafe { &*root };

		let old_thread_id = THREAD_ID.with(|x| x.replace(0));
		let old_worker = WORKER.with(|x| x.replace(false));
		let old_womanager = WOMANAGER.with(|x| x.replace(&raw const womanager.as_ref().unwrap().1));

		let old_waker = root.waker.fetch_or(1, AcqRel);
		atomic_wait_imp::wake_all(&root.waker);

		let idx;
		let womanager = 'register: {
			root.children_thread_requests.fetch_add(n_threads - 1, Relaxed);
			for (i, child) in root.children.iter().enumerate() {
				if AtomicArc::is_free(child) {
					match AtomicArc::compare_exchange_null(child, womanager.take().unwrap()) {
						Ok(womanager) => {
							idx = i;
							break 'register womanager;
						},
						Err(get) => womanager = Some(get),
					}
				}
			}

			root.children_thread_requests.fetch_sub(n_threads - 1, Relaxed);
			let __guard__ = defer(|| {
				WOMANAGER.with(|x| x.set(old_womanager));
				THREAD_ID.with(|x| x.set(old_thread_id));
				WORKER.with(|x| x.set(old_worker));
				root.waker.store(old_waker, Release);
			});
			return f();
		};
		let __guard__ = defer(|| {
			WOMANAGER.with(|x| x.set(old_womanager));
			THREAD_ID.with(|x| x.set(old_thread_id));
			WORKER.with(|x| x.set(old_worker));
			root.waker.store(old_waker, Release);

			let womanager = unsafe { &*womanager };

			womanager.1.waker.store(1, Release);
			womanager.1.announce_exit.store(true, Release);
			womanager.1.announce_rest.store(false, Release);
			atomic_wait_imp::wake_all(&womanager.1.waker);

			let places_not_filled = womanager.0.swap(0, AcqRel);
			root.children_thread_requests.fetch_sub(places_not_filled, Relaxed);

			barrier.wait();
			while root.children[idx].count.load(Acquire) > 1 {}
			assert!((womanager.1.worker.init.data.load(Relaxed) >> 16) & barrier::MASK == 1);
			assert!((barrier.init.data.load(Relaxed) >> 16) & barrier::MASK == 1);
			AtomicArc::clear(&root.children[idx]);
		});

		f()
	}
}

fn worker_loop(womanager: &Womanager) {
	unsafe {
		let root = ROOT.with(|x| x.get()).as_ref().unwrap_or(womanager);
		let thread_id;
		'register: loop {
			for (i, used) in womanager.worker.thread_id_used.iter().enumerate() {
				let val = used.load(Acquire);
				let trailing = val.trailing_ones();
				if trailing == 32 {
					continue;
				}

				let bit = 1u32 << trailing;
				let pos = 32 * i + trailing as usize;
				if pos < womanager.n_threads && used.fetch_or(bit, AcqRel) & bit == 0 {
					thread_id = pos;
					break 'register;
				}
			}
		}

		let old_womanager = WOMANAGER.with(|x| x.replace(womanager));
		let old_worker = WORKER.with(|x| x.replace(true));
		let old_root = ROOT.with(|x| x.replace(root));
		let old_thread_id = THREAD_ID.with(|x| x.replace(thread_id));

		let __guard__ = defer(|| {
			let trailing = thread_id % 32;
			let bit = 1u32 << trailing;
			womanager.worker.thread_id_used[thread_id / 32].fetch_and(!bit, AcqRel);

			WOMANAGER.with(|x| x.set(old_womanager));
			WORKER.with(|x| x.set(old_worker));
			ROOT.with(|x| x.set(old_root));
			THREAD_ID.with(|x| x.set(old_thread_id));
		});

		if !old_root.is_null() {
			assert!(old_root == &raw const *root);
		}

		let barrier = BarrierInit::barrier(&womanager.init, thread_id);

		if womanager.announce_exit.load(Acquire) {
			if core::ptr::eq(womanager, root) {
				root.n_spawn.fetch_sub(1, Relaxed);
			}
			barrier.exit();
			return;
		}

		'enter: loop {
			let barrier = BarrierInit::barrier(&womanager.worker.init, thread_id);

			let mut spin = 0;
			let max_pause = PAUSE_LIMIT.load(Relaxed);
			let max_spin = SPIN_LIMIT.load(Relaxed) / Ord::max(1, max_pause);

			'work: loop {
				if womanager.announce_exit.load(Acquire) {
					barrier.exit();
					break 'enter;
				}

				if !womanager.worker.task.load(Acquire).is_null() {
					spin = 0;

					thread_work(womanager);

					if barrier.wait_and_clear_while(&womanager.children_thread_requests, &womanager.worker.task) == ExitStatus::Exit {
						break 'work;
					}
				} else if womanager.children_thread_requests.load(Acquire) > 0 {
					barrier.exit();
					break 'work;
				} else {
					if spin < max_spin && !womanager.announce_rest.load(Acquire) {
						for _ in 0..max_pause {
							mock::spin_loop();
						}
						spin += 1;
					} else {
						atomic_wait_imp::wait(&womanager.waker, 0);
					}
				}
			}

			for child in womanager.children.iter() {
				if !AtomicArc::is_null(child) {
					if let Some(child) = AtomicArc::load(child) {
						if child.1.depth > womanager.depth {
							let (available_slots, child) = &*child;
							let available = available_slots.load(Acquire);
							if available > 0 {
								match available_slots.compare_exchange(available, available - 1, Release, Acquire) {
									Ok(_) => {
										assert!(womanager.children_thread_requests.fetch_sub(1, AcqRel) > 0);
										worker_loop(&*child);
									},
									Err(_) => {},
								}
							}
						}
					}
				}
			}
		}

		if core::ptr::eq(womanager, root) {
			root.n_spawn.fetch_sub(1, Relaxed);
		}
		barrier.exit();
	}
}

unsafe fn thread_work(womanager: &Womanager) {
	unsafe {
		let thread_id = THREAD_ID.with(|x| x.get());

		let n_threads = womanager.n_threads;
		let n_jobs = womanager.worker.n_jobs.load(Relaxed);
		assert!(!womanager.worker.task.load(Relaxed).is_null());
		let task = &**womanager.worker.task.load(Relaxed);
		let jobs_per_thread = n_jobs / n_threads;

		loop {
			let rem = womanager.worker.structured_jobs[thread_id].load(Acquire);
			if rem == 0 {
				break;
			}

			if womanager.worker.structured_jobs[thread_id]
				.compare_exchange(rem, rem - 1, Release, Relaxed)
				.is_ok()
			{
				let job = jobs_per_thread * thread_id + jobs_per_thread - rem;
				task(job, &womanager.worker.panic_slot);

				if rem == 1 {
					break;
				}
			}
		}

		loop {
			let mut max = 0;
			let mut argmax = 0;
			for other in 0..n_threads {
				let val = womanager.worker.structured_jobs[other].load(Acquire);
				if val > max {
					max = val;
					argmax = other;
				}
				if max == jobs_per_thread {
					break;
				}
			}
			if max == 0 {
				break;
			}

			let thread_id = argmax;

			if womanager.worker.structured_jobs[thread_id]
				.compare_exchange(max, max - 1, Release, Relaxed)
				.is_ok()
			{
				let job = jobs_per_thread * thread_id + jobs_per_thread - max;
				task(job, &womanager.worker.panic_slot);
			}
		}
		let mut rem = womanager.worker.unstructured_jobs.load(Acquire);
		while rem > 0 && rem != usize::MAX {
			match womanager.worker.unstructured_jobs.compare_exchange_weak(rem, rem - 1, Release, Acquire) {
				Ok(_) => {
					let job = n_jobs - rem;
					task(job, &womanager.worker.panic_slot);
					if rem == 1 {
						womanager.worker.unstructured_jobs.store(usize::MAX, Release);
					}
				},
				Err(new) => rem = new,
			}
		}
	}
}

fn for_each_raw_imp(n_jobs: usize, task: &(dyn Sync + Fn(usize))) {
	if n_jobs == 1 {
		return task(0);
	}

	unsafe {
		let root = ROOT.with(|x| x.get()).as_ref();
		match root {
			None => (0..n_jobs).into_par_iter().for_each(task),
			Some(root) => {
				if WORKER.with(|x| x.get()) {
					with_lock(n_jobs, || {
						for_each_raw_imp(n_jobs, task);
					});
				} else {
					WORKER.with(|x| x.set(true));
					let __guard__ = defer(|| {
						WORKER.with(|x| x.set(false));
					});

					let womanager = &*WOMANAGER.with(|x| x.get());
					#[cfg(not(any(loom, miri)))]
					if let Some(scope) = root.scope {
						let mut n_spawn = root.n_spawn.load(Acquire);
						while n_spawn < root.n_threads {
							match root.n_spawn.compare_exchange(n_spawn, n_spawn + 1, Release, Acquire) {
								Ok(_) => {
									scope.spawn(|_| (root.spawn)(root));
								},
								Err(new) => n_spawn = new,
							}
						}
					}
					#[derive(Copy, Clone, Debug)]
					struct AssertUnwindSafe<T>(T);
					impl<T> UnwindSafe for AssertUnwindSafe<T> {}
					impl<T> RefUnwindSafe for AssertUnwindSafe<T> {}

					let task = AssertUnwindSafe(task);

					let task = |thread_id: usize, panic_slot: &AtomicPtr<PanicLoad>| match std::panic::catch_unwind(|| {
						({ task }.0)(thread_id);
					}) {
						Ok(_) => {},
						Err(panic_load) => {
							let ptr = panic_slot.swap(null_mut(), AcqRel);
							if !ptr.is_null() {
								ptr.write(panic_load);
							}
						},
					};

					let mut storage = MaybeUninit::uninit();
					let mut task = (&raw const task) as *const TaskInner;
					{
						womanager.waker.store(1, Relaxed);
						let n_threads = womanager.n_threads;

						let (div, rem) = (n_jobs / n_threads, n_jobs % n_threads);

						for job in &womanager.worker.structured_jobs {
							job.store(div, Relaxed);
						}
						womanager.worker.n_jobs.store(n_jobs, Relaxed);
						womanager.worker.unstructured_jobs.store(rem, Relaxed);
						womanager.worker.panic_slot.store(storage.as_mut_ptr(), Relaxed);
						womanager.announce_rest.store(false, Relaxed);

						womanager.worker.task.store(&raw mut task, Release);
						atomic_wait_imp::wake_all(&womanager.waker);

						thread_work(womanager);
						womanager.worker.leader.wait_and_clear(&womanager.worker.task);
						womanager.waker.store(0, Relaxed);

						if womanager.worker.panic_slot.load(Relaxed).is_null() {
							std::panic::resume_unwind(storage.assume_init());
						}
					}
				}
			},
		}
	}
}

fn for_each_imp<I: IntoParallelIterator<Iter: IndexedParallelIterator>>(n_jobs: usize, iter: I, f: impl Sync + Fn(I::Item)) {
	struct C<'a, F>(&'a F, usize, usize);
	impl<T, F: Sync + Fn(T)> ProducerCallback<T> for C<'_, F> {
		type Output = ();

		fn callback<P>(self, mut producer: P) -> Self::Output
		where
			P: Producer<Item = T>,
		{
			let len = self.1;
			let n_jobs = self.2;

			let mut v = Vec::with_capacity(len);

			let div = len / n_jobs;
			let rem = len % n_jobs;

			for _ in 0..rem {
				let left;
				(left, producer) = producer.split_at(div + 1);
				v.push(SyncCell(UnsafeCell::new(Some(left))));
			}
			for _ in rem..n_jobs {
				let left;
				(left, producer) = producer.split_at(div);
				v.push(SyncCell(UnsafeCell::new(Some(left))));
			}

			let f = self.0;

			for_each_raw_imp(n_jobs, &|idx: usize| unsafe {
				#[cfg(not(loom))]
				let p = (&mut *v[idx].0.get()).take().unwrap();

				#[cfg(loom)]
				let p = v[idx].0.with_mut(|x| (*x).take().unwrap());
				p.into_iter().for_each(f);
			});
		}
	}

	let iter = iter.into_par_iter();
	let len = iter.len();
	let n_jobs = Ord::min(len, n_jobs);
	iter.with_producer(C(&f, len, n_jobs));
}

pub fn relieve_workers() {
	if !WOMANAGER.with(|x| x.get()).is_null() {
		unsafe { (*WOMANAGER.with(|x| x.get())).announce_rest.store(true, Release) }
	}
}

pub fn for_each<T, I: IntoParallelIterator<Iter: IndexedParallelIterator, Item = T>>(n_jobs: usize, iter: I, f: impl Sync + Fn(T)) {
	for_each_imp(n_jobs, iter.into_par_iter(), f);
}

pub fn for_each_raw(n_jobs: usize, f: impl Sync + Fn(usize)) {
	for_each_raw_imp(n_jobs, (&f) as &(dyn Sync + Fn(usize)));
}

#[cfg(test)]
mod tests {
	#![allow(non_snake_case)]

	use super::*;
	use aligned_vec::avec;

	#[test]
	fn test_lock() {
		let f = || {
			let n_jobs = 8;
			with_lock(n_jobs, || {
				for_each(2, [(), ()], |()| {});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 10000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	const REPEAT: usize = 4096;

	#[test]
	fn test_nested_for_each() {
		let f = || {
			let m = 16;
			let n = 16;

			let A = &mut *avec![0.0; m * n];
			let n_jobs = 16;

			A.fill(0.0);
			with_lock(n_jobs, || {
				for _ in 0..n {
					for_each(4, A.par_chunks_mut(m * n.div_ceil(n_jobs)), |cols| {
						for_each(4, cols.par_chunks_mut(m), |col| {
							for e in col {
								*e += 1.0;
							}
						});
					});
				}
			});

			for e in &*A {
				assert_eq!(*e, n as f64);
			}
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 10000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		for _ in 0..REPEAT {
			f();
		}
	}

	#[test]
	fn test_nested_lock() {
		let f = || {
			let mut A = [1; 128];
			with_lock(16, || {
				with_lock(4, || {
					for_each(4, &mut A, |x| {
						*x = 0;
					});
				});
			});
			assert_eq!(A, [0; 128]);
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 10000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn test_scope_recursive() {
		let f = || {
			let m = 16;
			let n = 16;

			SPIN_LIMIT.store(0, Relaxed);

			let A = &mut *avec![0.0; m * n];
			let B = &mut *avec![0.0; m * n];
			let n_jobs = 16;

			with_lock(16, || {
				let A = [&mut *A, &mut *B];
				let len = A.len();
				for_each(len, A.into_par_iter(), |A| {
					A.fill(0.0);
					with_lock(1, || {
						for _ in 0..n {
							for_each(n_jobs, A.par_chunks_mut(m * n / n_jobs), |_| {
								with_lock(16, || {});
							});
						}
					});
				});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 10000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn par_scope_coarse() {
		let f = || {
			let m = 16;
			let n = 16;

			let A = &mut *avec![0.0; m * n];
			let B = &mut *avec![0.0; m * n];
			let n_jobs = 8;

			with_lock(16, || {
				for_each(2, [&mut *A, &mut *B], |A| {
					A.fill(0.0);
					with_lock(n_jobs, || {
						for _ in 0..n {
							for_each(n_jobs / 2, A.par_chunks_mut(m * n / n_jobs), |cols| {
								for col in cols.chunks_mut(m) {
									for e in col {
										*e += 1.0;
									}
								}
							});
						}
					});
					for e in &*A {
						assert_eq!(*e, n as f64);
					}
				});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 10000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}
}

#[cfg(all(loom, test))]
mod loom_tests {
	#![allow(non_snake_case)]

	use super::*;
	use aligned_vec::avec;

	#[test]
	fn lock() {
		let f = || {
			let n_jobs = 8;
			with_lock(n_jobs, || {
				for_each(2, [(), ()], |()| {});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 100000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn nested_for_each() {
		let f = || {
			with_lock(4, || {
				for_each(2, [(); 4], |()| {
					with_lock(4, || {
						for_each(4, [(); 4], |()| {});
					});
				});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 100000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn nested_lock() {
		let f = || {
			let mut A = [1; 128];
			with_lock(16, || {
				with_lock(4, || {
					for_each(4, &mut A, |x| {
						*x = 0;
					});
				});
			});
			assert_eq!(A, [0; 128]);
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 100000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn scope_recursive() {
		let f = || {
			let m = 16;
			let n = 16;

			SPIN_LIMIT.store(0, Relaxed);

			let A = &mut *avec![0.0; m * n];
			let B = &mut *avec![0.0; m * n];
			let n_jobs = 16;

			with_lock(16, || {
				let A = [&mut *A, &mut *B];
				let len = A.len();
				for_each(len, A.into_par_iter(), |A| {
					A.fill(0.0);
					with_lock(1, || {
						for _ in 0..n {
							for_each(n_jobs, A.par_chunks_mut(m * n / n_jobs), |_| {
								with_lock(16, || {});
							});
						}
					});
				});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 100000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}

	#[test]
	fn par_scope_coarse() {
		let f = || {
			let m = 16;
			let n = 16;

			let A = &mut *avec![0.0; m * n];
			let B = &mut *avec![0.0; m * n];
			let n_jobs = 8;

			with_lock(16, || {
				for_each(2, [&mut *A, &mut *B], |A| {
					A.fill(0.0);
					with_lock(n_jobs, || {
						for _ in 0..n {
							for_each(n_jobs / 2, A.par_chunks_mut(m * n / n_jobs), |cols| {
								for col in cols.chunks_mut(m) {
									for e in col {
										*e += 1.0;
									}
								}
							});
						}
					});
					for e in &*A {
						assert_eq!(*e, n as f64);
					}
				});
			});
		};
		#[cfg(loom)]
		{
			let mut builder = loom::model::Builder::new();
			builder.max_branches = 100000;
			builder.check(f);
		}
		#[cfg(not(loom))]
		f();
	}
}
