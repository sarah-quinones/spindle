use super::*;
use equator::assert;

#[derive(Debug)]
pub struct BarrierInit {
    /// bit layout:
    /// - `0..15`: count
    /// - `16..31`: max
    /// - `31`: global sense
    pub data: AtomicU32,
}

pub const MASK: u32 = (1u32 << 15) - 1;

#[derive(Debug)]
pub struct Barrier {
    pub init: Arc<BarrierInit>,
    // unsound but it's an internal api so.. eh
    local_sense: SyncUnsafeCell<bool>,
    thread_id: usize,
}

impl BarrierInit {
    #[inline]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            data: AtomicU32::new(0),
        })
    }

    #[inline]
    pub fn barrier(this: &Arc<Self>, thread_id: usize) -> Barrier {
        loop {
            let data = this.data.load(Acquire);
            let max = (data >> 16) & MASK;
            if max == MASK {
                panic!();
            }

            if data & MASK == 0 && max > 0 {
                continue;
            }

            match this
                .data
                .compare_exchange(data, data + (1u32 | (1u32 << 16)), Release, Relaxed)
            {
                Ok(_) => {
                    return Barrier {
                        local_sense: SyncUnsafeCell::new(data >> 31 != 0),
                        init: this.clone(),
                        thread_id,
                    };
                }
                _ => {}
            }

            mock::spin_loop();
        }
    }

    #[inline]
    pub fn current_num_threads(&self) -> usize {
        ((self.data.load(Relaxed) >> 16) & MASK) as usize
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Exit<'a> {
    Exit,
    Remain,
    ExitIfPositive(&'a AtomicUsize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExitStatus {
    Exit,
    Remain,
}

impl Barrier {
    #[inline(always)]
    fn wait_imp(&self, ptr: Option<&AtomicPtr<*const TaskInner>>, exit: Exit) -> ExitStatus {
        #[cfg(loom)]
        let local_sense = !self.local_sense.0.with(|x| unsafe { *x });
        #[cfg(loom)]
        self.local_sense.0.with_mut(|x| unsafe { *x = local_sense });

        #[cfg(not(loom))]
        let local_sense = !(unsafe { *self.local_sense.0.get() });
        #[cfg(not(loom))]
        unsafe {
            *self.local_sense.0.get() = local_sense
        };

        let dec = if matches!(exit, Exit::Exit) {
            1 | (1 << 16)
        } else {
            1
        };

        let data = self.init.data.fetch_sub(dec, AcqRel);
        let count = data & MASK;
        if count == 0 {
            dbg!((&raw const self.init.data, self.thread_id));
        }
        assert!(count > 0);

        if count == 1 {
            let mut max = (data >> 16) & MASK;
            if matches!(exit, Exit::Exit) {
                max -= 1;
            };

            if let Some(ptr) = ptr {
                ptr.store(null_mut(), Release);
            }

            let val = max | (max << 16) | ((local_sense as u32) << 31);
            self.init.data.store(val, Release);
            atomic_wait_imp::wake_all(&self.init.data);

            if matches!(exit, Exit::Exit) {
                ExitStatus::Exit
            } else {
                ExitStatus::Remain
            }
        } else if matches!(exit, Exit::Exit) {
            ExitStatus::Exit
        } else {
            let mut spin = 0u32;
            let max_pause = PAUSE_LIMIT.load(Relaxed);
            let max_spin = SPIN_LIMIT.load(Relaxed) / Ord::max(1, max_pause);

            loop {
                let data = self.init.data.load(Acquire);
                if data >> 31 == local_sense as u32 {
                    return ExitStatus::Remain;
                }

                if let Exit::ExitIfPositive(exit) = exit {
                    if exit.load(Acquire) > 0
                        && self
                            .init
                            .data
                            .compare_exchange(data, data - (1 << 16), Release, Relaxed)
                            .is_ok()
                    {
                        return ExitStatus::Exit;
                    }
                }

                if spin < max_spin {
                    for _ in 0..max_pause {
                        mock::spin_loop();
                    }
                    spin += 1;
                } else {
                    atomic_wait_imp::wait(&self.init.data, data);
                }
            }
        }
    }

    #[inline(never)]
    pub fn wait(&self) {
        self.wait_imp(None, Exit::Remain);
    }

    #[inline(never)]
    pub fn wait_and_clear(&self, ptr: &AtomicPtr<*const TaskInner>) {
        self.wait_imp(Some(ptr), Exit::Remain);
    }

    #[inline(never)]
    pub fn exit(&self) {
        self.wait_imp(None, Exit::Exit);
    }

    // #[inline(never)]
    // pub fn exit_and_clear(&self, ptr: &AtomicPtr<*const TaskInner>) {
    //     self.wait_imp(Some(ptr), Exit::Exit);
    // }

    // #[inline(never)]
    // pub fn wait_while(&self, zero: &AtomicUsize) -> ExitStatus {
    //     self.wait_imp(None, Exit::ExitIfPositive(zero))
    // }

    #[inline(never)]
    pub fn wait_and_clear_while(
        &self,
        zero: &AtomicUsize,
        ptr: &AtomicPtr<*const TaskInner>,
    ) -> ExitStatus {
        self.wait_imp(Some(ptr), Exit::ExitIfPositive(zero))
    }
}

unsafe impl Sync for Barrier {}
unsafe impl Send for Barrier {}
