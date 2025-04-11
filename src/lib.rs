use barrier::Barrier;
use barrier::BarrierInit;
use barrier::ExitStatus;
use rayon::iter::plumbing::Producer;
use rayon::iter::plumbing::ProducerCallback;
use std::any::Any;
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::iter;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::panic::RefUnwindSafe;
use std::panic::UnwindSafe;
use std::ptr::{null, null_mut};
use std::sync::atomic::*;
use std::sync::Arc;
use Ordering::*;

use equator::assert;

use aarc::AtomicArc;
use rayon::prelude::*;

#[cfg(not(miri))]
type ThdScope<'a, 'b> = &'a rayon::Scope<'b>;
#[cfg(miri)]
type ThdScope<'a, 'b> = &'a std::thread::Scope<'a, 'b>;

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
struct SyncUnsafeCell<T: ?Sized>(UnsafeCell<T>);

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

    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0.get()
    }
}

pub const DEFAULT_SPIN_LIMIT: u32 = 65536;
pub const DEFAULT_PAUSE_LIMIT: u32 = 8;

pub const SPIN_LIMIT: AtomicU32 = AtomicU32::new(DEFAULT_SPIN_LIMIT);
pub const PAUSE_LIMIT: AtomicU32 = AtomicU32::new(DEFAULT_PAUSE_LIMIT);

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
        let thread_id_used = iter::repeat_n(0, n_threads.div_ceil(32))
            .map(AtomicU32::new)
            .collect::<Box<[_]>>();

        thread_id_used[0].store(1, Relaxed);
        Self {
            thread_id_used,
            task: AtomicPtr::new(null_mut()),
            panic_slot: AtomicPtr::new(null_mut()),
            n_jobs: AtomicUsize::new(0),
            structured_jobs: iter::repeat_n(0, n_threads).map(AtomicUsize::new).collect(),
            unstructured_jobs: AtomicUsize::new(0),
            leader: init.barrier(0),
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

thread_local! {
    static WORKER: Cell<bool> = const { Cell::new(true) };
    static WOMANAGER: Cell<*const Womanager> = const { Cell::new(null()) };
    static ROOT: Cell<*const Womanager> = const { Cell::new(null()) };
    static THREAD_ID: Cell<usize> = const { Cell::new(0) };
}

pub fn current_num_threads() -> usize {
    let root = WOMANAGER.get();
    if !root.is_null() {
        unsafe { &*root }.worker.init.current_num_threads()
    } else {
        rayon::current_num_threads()
    }
}

pub fn max_num_threads() -> usize {
    let root = WOMANAGER.get();
    if !root.is_null() {
        unsafe { &*root }.n_threads
    } else {
        rayon::current_num_threads()
    }
}

pub fn current_thread_index() -> Option<usize> {
    let root = WOMANAGER.get();
    if !root.is_null() {
        if WORKER.get() {
            Some(THREAD_ID.get())
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
    let n_threads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(16);

    #[cfg(not(miri))]
    let n_threads = rayon::current_num_threads();

    let n_threads = Ord::min(n_threads, max_threads);
    let root = ROOT.get();

    let n_spawn = &AtomicUsize::new(0);
    let spawn: &(dyn Sync + Fn(&Womanager)) = &|womanager: &Womanager| worker_loop(womanager);
    let mut womanager = Womanager {
        n_threads,
        waker: AtomicU32::new(0),
        spawn: unsafe { core::mem::transmute(spawn) },
        scope: None,
        n_spawn: unsafe { &*(n_spawn as *const _) },
        worker: Worker::new(n_threads),
        children: iter::repeat_n((), n_threads)
            .map(|()| AtomicArc::new(None))
            .collect(),
        children_thread_requests: AtomicUsize::new(0),
        init: BarrierInit::new(),
        announce_exit: AtomicBool::new(false),
        announce_rest: AtomicBool::new(true),
        depth: unsafe { WOMANAGER.get().as_ref().map(|w| w.depth + 1).unwrap_or(0) },
    };

    if root.is_null() {
        #[cfg(not(miri))]
        use rayon::scope;
        #[cfg(miri)]
        use std::thread::scope;

        scope(|scope| {
            womanager.scope = Some(unsafe { &*(scope as *const _ as *const _) });

            let barrier = womanager.init.barrier(0);
            womanager.n_spawn.store(n_threads, Relaxed);
            for _ in 1..n_threads {
                #[cfg(not(miri))]
                scope.spawn(|_| (womanager.spawn)(&womanager));
                #[cfg(miri)]
                scope.spawn(|| (womanager.spawn)(&womanager));
            }

            let old_root = ROOT.replace(&raw const womanager);
            let old_womanager = WOMANAGER.replace(&raw const womanager);
            let old_thread_id = THREAD_ID.replace(0);
            let old_worker = WORKER.replace(false);

            let __guard__ = defer(|| {
                womanager.waker.store(1, Release);
                womanager.announce_exit.store(true, Release);
                womanager.announce_rest.store(false, Release);
                atomic_wait::wake_all(&womanager.waker);
                ROOT.set(old_root);
                THREAD_ID.set(old_thread_id);
                WORKER.set(old_worker);
                WOMANAGER.set(old_womanager);
                barrier.wait();
            });

            f()
        })
    } else {
        if !WORKER.get() {
            return f();
        }

        let womanager = aarc::Arc::new((AtomicUsize::new(n_threads - 1), womanager));
        let root = unsafe { &*root };

        let old_thread_id = THREAD_ID.replace(0);
        let old_worker = WORKER.replace(false);
        let old_womanager = WOMANAGER.replace(&raw const womanager.1);

        let old_waker = root.waker.fetch_or(1, AcqRel);
        atomic_wait::wake_all(&root.waker);

        let idx;
        'register: {
            root.children_thread_requests
                .fetch_add(n_threads - 1, Relaxed);
            for (i, child) in root.children.iter().enumerate() {
                if child.load().is_none() {
                    if child.compare_exchange(null(), Some(&womanager)).is_ok() {
                        idx = i;
                        break 'register;
                    }
                }
            }

            root.children_thread_requests
                .fetch_sub(n_threads - 1, Relaxed);
            let __guard__ = defer(|| {
                WOMANAGER.set(old_womanager);
                THREAD_ID.set(old_thread_id);
                WORKER.set(old_worker);
                root.waker.store(old_waker, Release);
            });
            return f();
        }

        let barrier = womanager.1.init.barrier(0);
        let __guard__ = defer(|| {
            WOMANAGER.set(old_womanager);
            THREAD_ID.set(old_thread_id);
            WORKER.set(old_worker);
            root.waker.store(old_waker, Release);

            womanager.1.waker.store(1, Release);
            womanager.1.announce_exit.store(true, Release);
            womanager.1.announce_rest.store(false, Release);
            atomic_wait::wake_all(&womanager.1.waker);
            let child = &root.children[idx];

            {
                let c = child.load().unwrap();
                let places_not_filled = c.0.swap(usize::MAX, AcqRel);
                root.children_thread_requests
                    .fetch_sub(places_not_filled, Relaxed);
            }
            child.store(None::<&aarc::Arc<_>>);

            barrier.wait();
        });

        f()
    }
}

fn worker_loop(womanager: &Womanager) {
    unsafe {
        let root = ROOT.get().as_ref().unwrap_or(womanager);
        let thread_id;
        'register: loop {
            for (i, used) in womanager.worker.thread_id_used.iter().enumerate() {
                let val = used.load(Acquire);
                let trailing = val.trailing_ones();
                let bit = 1u32 << trailing;
                if trailing < 32 {
                    if used.fetch_or(bit, AcqRel) & bit == 0 {
                        thread_id = 32 * i + trailing as usize;
                        break 'register;
                    }
                }
            }
        }

        let old_womanager = WOMANAGER.replace(womanager);
        let old_worker = WORKER.replace(true);
        let old_root = ROOT.replace(root);
        let old_thread_id = THREAD_ID.replace(thread_id);

        let __guard__ = defer(|| {
            let trailing = thread_id % 32;
            let bit = 1u32 << trailing;
            womanager.worker.thread_id_used[thread_id / 32].fetch_and(!bit, AcqRel);

            WOMANAGER.set(old_womanager);
            WORKER.set(old_worker);
            ROOT.set(old_root);
            THREAD_ID.set(old_thread_id);
        });

        if !old_root.is_null() {
            assert!(old_root == &raw const *root);
        }

        let barrier = womanager.init.barrier(thread_id);

        if womanager.announce_exit.load(Acquire) {
            if core::ptr::eq(womanager, root) {
                root.n_spawn.fetch_sub(1, Relaxed);
            }
            barrier.exit();
            return;
        }

        'enter: loop {
            let barrier = womanager.worker.init.barrier(thread_id);

            let mut spin = 0;
            let max_pause = PAUSE_LIMIT.load(Relaxed);
            let max_spin = SPIN_LIMIT.load(Relaxed);

            'work: loop {
                if womanager.announce_exit.load(Acquire) {
                    barrier.exit();
                    break 'enter;
                }

                if !womanager.worker.task.load(Acquire).is_null() {
                    spin = 0;

                    thread_work(womanager);

                    if barrier.wait_and_clear_while(
                        &womanager.children_thread_requests,
                        &womanager.worker.task,
                    ) == ExitStatus::Exit
                    {
                        break 'work;
                    }
                } else if womanager.children_thread_requests.load(Acquire) > 0 {
                    barrier.exit();
                    break 'work;
                } else {
                    if spin < max_spin && !womanager.announce_rest.load(Acquire) {
                        for _ in 0..max_pause {
                            core::hint::spin_loop();
                        }
                        spin += 1;
                    } else {
                        atomic_wait::wait(&womanager.waker, 0);
                    }
                }
            }

            for child in womanager.children.iter() {
                if let Some(child) = child.load() {
                    if child.1.depth > womanager.depth {
                        let (available_slots, child) = &*child;
                        let available = available_slots.load(Acquire);
                        if available > 0 && available != usize::MAX {
                            match available_slots.compare_exchange(
                                available,
                                available - 1,
                                Release,
                                Acquire,
                            ) {
                                Ok(_) => {
                                    assert!(
                                        womanager.children_thread_requests.fetch_sub(1, AcqRel) > 0
                                    );
                                    worker_loop(&*child);
                                }
                                Err(_) => {}
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
        let thread_id = THREAD_ID.get();

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
            match womanager.worker.unstructured_jobs.compare_exchange_weak(
                rem,
                rem - 1,
                Release,
                Acquire,
            ) {
                Ok(_) => {
                    let job = n_jobs - rem;
                    task(job, &womanager.worker.panic_slot);
                    if rem == 1 {
                        womanager
                            .worker
                            .unstructured_jobs
                            .store(usize::MAX, Release);
                    }
                }
                Err(new) => rem = new,
            }
        }
    }
}

fn for_each_raw_imp(n_jobs: usize, task: &(dyn Sync + Fn(usize))) {
    unsafe {
        let root = ROOT.get().as_ref();
        match root {
            None => (0..n_jobs).into_par_iter().for_each(task),
            Some(root) => {
                if WORKER.get() {
                    with_lock(max_num_threads(), || {
                        for_each_raw_imp(n_jobs, task);
                    });
                } else {
                    WORKER.set(true);
                    let __guard__ = defer(|| {
                        WORKER.set(false);
                    });

                    let womanager = &*WOMANAGER.get();
                    if let Some(scope) = root.scope {
                        let mut n_spawn = root.n_spawn.load(Acquire);
                        while n_spawn < root.n_threads {
                            match root.n_spawn.compare_exchange(
                                n_spawn,
                                n_spawn + 1,
                                Release,
                                Acquire,
                            ) {
                                Ok(_) => {
                                    #[cfg(not(miri))]
                                    scope.spawn(|_| (root.spawn)(root));
                                    #[cfg(miri)]
                                    scope.spawn(|| (root.spawn)(root));
                                }
                                Err(new) => n_spawn = new,
                            }
                        }
                    }
                    #[derive(Copy, Clone, Debug)]
                    struct AssertUnwindSafe<T>(T);
                    impl<T> UnwindSafe for AssertUnwindSafe<T> {}
                    impl<T> RefUnwindSafe for AssertUnwindSafe<T> {}

                    let task = AssertUnwindSafe(task);

                    let task = |thread_id: usize, panic_slot: &AtomicPtr<PanicLoad>| {
                        match std::panic::catch_unwind(|| {
                            ({ task }.0)(thread_id);
                        }) {
                            Ok(_) => {}
                            Err(panic_load) => {
                                let ptr = panic_slot.swap(null_mut(), AcqRel);
                                if !ptr.is_null() {
                                    ptr.write(panic_load);
                                }
                            }
                        }
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
                        womanager
                            .worker
                            .panic_slot
                            .store(storage.as_mut_ptr(), Relaxed);
                        womanager.announce_rest.store(false, Relaxed);

                        womanager.worker.task.store(&raw mut task, Release);
                        atomic_wait::wake_all(&womanager.waker);

                        thread_work(womanager);
                        womanager
                            .worker
                            .leader
                            .wait_and_clear(&womanager.worker.task);
                        womanager.waker.store(0, Relaxed);

                        if womanager.worker.panic_slot.load(Relaxed).is_null() {
                            std::panic::resume_unwind(storage.assume_init());
                        }
                    }
                }
            }
        }
    }
}

fn for_each_imp<I: IntoParallelIterator<Iter: IndexedParallelIterator>>(
    n_jobs: usize,
    iter: I,
    f: impl Sync + Fn(I::Item),
) {
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
                let p = (&mut *v[idx].0.get()).take().unwrap();
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
    if !WOMANAGER.get().is_null() {
        unsafe { (*WOMANAGER.get()).announce_rest.store(true, Release) }
    }
}

pub fn for_each<T, I: IntoParallelIterator<Iter: IndexedParallelIterator, Item = T>>(
    n_jobs: usize,
    iter: I,
    f: impl Sync + Fn(T),
) {
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
    fn test_nested_for_each() {
        let m = 16;
        let n = 16;

        let A = &mut *avec![0.0; m * n];
        let n_jobs = 8;

        A.fill(0.0);
        with_lock(n_jobs, || {
            for _ in 0..n {
                for_each(2, A.par_chunks_mut(m * n.div_ceil(n_jobs)), |cols| {
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
    }

    #[test]
    fn test_nested_lock() {
        let mut A = [1; 128];
        with_lock(16, || {
            with_lock(4, || {
                for_each(4, &mut A, |x| {
                    *x = 0;
                });
            });
        });
        assert_eq!(A, [0; 128]);
    }

    #[test]
    fn test_scope_recursive() {
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
    }

    #[test]
    fn par_scope_coarse() {
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
    }
}
