use std::{
    cell::{Ref, RefMut},
    error::Error,
    u16,
};

use crate::{SilenceMask, MAX_BLOCK_FRAMES};

pub trait AudioNode: 'static {
    fn info(&self) -> AudioNodeInfo;

    /// Activate the audio node for processing.
    fn activate(
        &mut self,
        sample_rate: f64,
        num_inputs: u16,
        num_outputs: u16,
    ) -> Result<Box<dyn AudioNodeProcessor>, Box<dyn Error>>;

    /// Called when the processor counterpart has been deactivated
    /// and dropped.
    #[allow(unused)]
    fn deactivate(&mut self) {}
}

pub trait AudioNodeProcessor: 'static + Send {
    /// Process the given block of audio. Only process data in the
    /// buffers up to `frames`.
    ///
    /// Note, all output buffers *MUST* be filled with data up to
    /// `frames`.
    ///
    /// If any output buffers contain all zeros up to `frames` (silent),
    /// then mark that buffer as silent in [`ProcInfo::out_silence_mask`].
    fn process(
        &mut self,
        frames: usize,
        proc_info: ProcInfo,
        inputs: &[Ref<[f32; MAX_BLOCK_FRAMES]>],
        outputs: &mut [RefMut<[f32; MAX_BLOCK_FRAMES]>],
    ) -> ProcessStatus;
}

/// Additional information about an [`AudioNode`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioNodeInfo {
    /// The minimum number of input buffers this node supports
    pub num_min_supported_inputs: u16,
    /// The maximum number of input buffers this node supports
    ///
    /// This value must be less than `64`.
    pub num_max_supported_inputs: u16,

    /// The minimum number of output buffers this node supports
    pub num_min_supported_outputs: u16,
    /// The maximum number of output buffers this node supports
    ///
    /// This value must be less than `64`.
    pub num_max_supported_outputs: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessStatus {
    Ok,
    Err { msg: &'static str },
}

/// Additional information for processing audio
pub struct ProcInfo<'a> {
    /// An optional optimization hint on which input channels contain
    /// all zeros (silence). The first bit (`0x1`) is the first channel,
    /// the second bit is the second channel, and so on.
    pub in_silence_mask: SilenceMask,

    /// An optional optimization hint to notify the host which output
    /// channels contain all zeros (silence). The first bit (`0x1`) is
    /// the first channel, the second bit is the second channel, and so
    /// on.
    ///
    /// By default no channels are flagged as silent.
    pub out_silence_mask: &'a mut SilenceMask,
}

pub struct DummyAudioNode;

impl AudioNode for DummyAudioNode {
    fn info(&self) -> AudioNodeInfo {
        AudioNodeInfo {
            num_min_supported_inputs: 0,
            num_max_supported_inputs: u16::MAX,
            num_min_supported_outputs: 0,
            num_max_supported_outputs: u16::MAX,
        }
    }

    /// Activate the audio node for processing.
    fn activate(
        &mut self,
        _sample_rate: f64,
        _num_inputs: u16,
        _num_outputs: u16,
    ) -> Result<Box<dyn AudioNodeProcessor>, Box<dyn Error>> {
        Ok(Box::new(DummyAudioNodeProcessor))
    }
}

pub struct DummyAudioNodeProcessor;

impl AudioNodeProcessor for DummyAudioNodeProcessor {
    fn process(
        &mut self,
        _frames: usize,
        _proc_info: ProcInfo,
        _inputs: &[Ref<[f32; MAX_BLOCK_FRAMES]>],
        _outputs: &mut [RefMut<[f32; MAX_BLOCK_FRAMES]>],
    ) -> ProcessStatus {
        ProcessStatus::Ok
    }
}
