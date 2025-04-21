use crate::mock::{
    atomic::{AtomicPtr, Ordering::*},
    Arc,
};
use std::{ops::Deref, ptr::null_mut, sync::atomic::AtomicUsize};

#[repr(C)]
pub struct AtomicArc<T> {
    raw: AtomicPtr<T>,
    count: AtomicUsize,
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
        let old = this.count.fetch_add(1, AcqRel);
        if old == 0 {
            this.count.fetch_sub(1, Release);
            return None;
        }

        let raw = this.raw.load(Acquire);
        if raw.is_null() {
            this.count.fetch_sub(1, Release);
            return None;
        }

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
                if self.count.fetch_sub(1, Release) == 1 {
                    unsafe {
                        Arc::decrement_strong_count(self.ptr);
                    }
                }
            }
        }

        Some(Guard {
            ptr: raw,
            count: &this.count,
        })
    }

    #[inline(never)]
    pub fn clear(this: &Self) {
        let ptr = this.raw.load(Acquire);
        this.raw.store(null_mut(), Release);

        if !ptr.is_null() {
            if this.count.fetch_sub(1, Release) == 1 {
                unsafe { Arc::decrement_strong_count(ptr) };
            }
        }
    }

    #[inline(never)]
    pub fn compare_exchange_null(this: &Self, new: Arc<T>) -> Result<*const T, Arc<T>> {
        if this
            .count
            .compare_exchange_weak(0, 1, AcqRel, Relaxed)
            .is_err()
        {
            return Err(new);
        }

        let ptr = Arc::into_raw(new) as *mut T;
        this.raw.store(ptr, Relaxed);

        Ok(ptr)
    }
}
