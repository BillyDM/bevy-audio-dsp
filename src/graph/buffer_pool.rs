use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Range,
};

use super::schedule::BufferId;

#[derive(Clone, Copy)]
pub struct BlockBuffer<const MAX_BLOCK_SIZE: usize> {
    pub data: [f32; MAX_BLOCK_SIZE],
    pub is_silent: bool,
}

impl<const MAX_BLOCK_SIZE: usize> BlockBuffer<MAX_BLOCK_SIZE> {
    #[inline(always)]
    pub fn range(&self, range: BlockRange<MAX_BLOCK_SIZE>) -> &[f32] {
        // SAFETY:
        //
        // The constructor for `BlockRange` ensures that the values are
        // within bounds.
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(range.0.start as usize),
                (range.0.end - range.0.start) as usize,
            )
        }
    }

    #[inline(always)]
    pub fn range_mut(&mut self, range: BlockRange<MAX_BLOCK_SIZE>) -> &mut [f32] {
        // SAFETY:
        //
        // The constructor for `BlockRange` ensures that the values are
        // within bounds.
        //
        // `self` is borrowed as mutable which ensures that borrow rules
        // are properly followed.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(range.0.start as usize),
                (range.0.end - range.0.start) as usize,
            )
        }
    }
}

impl<const MAX_BLOCK_SIZE: usize> Default for BlockBuffer<MAX_BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            data: [0.0; MAX_BLOCK_SIZE],
            is_silent: true,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockFrames<const MAX_BLOCK_SIZE: usize>(u32);

impl<const MAX_BLOCK_SIZE: usize> BlockFrames<MAX_BLOCK_SIZE> {
    pub fn new(frames: u32) -> Option<Self> {
        if frames <= MAX_BLOCK_SIZE as u32 {
            Some(Self(frames))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get(&self) -> usize {
        if self.0 <= MAX_BLOCK_SIZE as u32 {
            self.0 as usize
        } else {
            // SAFETY: The constructor ensures that the value is less
            // than or equal to MAX_BLOCK_SIZE.
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }
    }
}

impl<const MAX_BLOCK_SIZE: usize> From<BlockFrames<MAX_BLOCK_SIZE>> for usize {
    #[inline(always)]
    fn from(value: BlockFrames<MAX_BLOCK_SIZE>) -> Self {
        value.get()
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct BlockRange<const MAX_BLOCK_SIZE: usize>(Range<u32>);

impl<const MAX_BLOCK_SIZE: usize> BlockRange<MAX_BLOCK_SIZE> {
    pub fn new(range: Range<u32>) -> Option<Self> {
        if range.start >= MAX_BLOCK_SIZE as u32 || range.end > MAX_BLOCK_SIZE as u32 {
            None
        } else {
            Some(Self(range))
        }
    }

    /// Construct a new [`BlockRange`] without checking that the given
    /// value is within bounds.
    ///
    /// # SAFETY:
    /// All the following must be true for this to be safe:
    /// * `range.start < MAX_BLOCK_SIZE`
    /// * `range.end <= MAX_BLOCK_SIZE`
    pub unsafe fn new_unchecked(range: Range<u32>) -> Self {
        Self(range)
    }

    pub fn get(&self) -> Range<u32> {
        self.0.clone()
    }
}

pub(crate) struct BufferPool<const MAX_BLOCK_SIZE: usize> {
    buffers: Vec<RefCell<BlockBuffer<MAX_BLOCK_SIZE>>>,
}

impl<const MAX_BLOCK_SIZE: usize> BufferPool<MAX_BLOCK_SIZE> {
    pub fn new(num_buffers: usize) -> Self {
        Self {
            buffers: (0..num_buffers)
                .map(|_| RefCell::new(BlockBuffer::default()))
                .collect(),
        }
    }

    pub fn buffer<'a>(&'a self, id: BufferId) -> Ref<'a, BlockBuffer<MAX_BLOCK_SIZE>> {
        RefCell::borrow(&self.buffers[id])
    }

    pub fn buffer_mut<'a>(&'a self, id: BufferId) -> RefMut<'a, BlockBuffer<MAX_BLOCK_SIZE>> {
        RefCell::borrow_mut(&self.buffers[id])
    }
}
