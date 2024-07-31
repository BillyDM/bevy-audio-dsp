use std::{
    hash::Hash,
    num::{NonZeroU32, NonZeroUsize},
    time::{Duration, Instant},
};

use crate::{graph::buffer_pool::BlockBuffer, BlockFrames, NormalVal, MAX_BLOCK_SIZE};

pub const MAX_IN_CHANNELS: usize = 4;
pub const MAX_OUT_CHANNELS: usize = 4;

pub type ParamID = u32;

/// A node which processes audio
pub trait AudioNode {
    /// Get the parameters of this node.
    ///
    /// This method will only be called once on initialization.
    fn parameters(&self) -> Vec<ParamInfo> {
        Vec::new()
    }

    /// Get the channel configuration of this node.
    ///
    /// This method will only be called once on initialization.
    fn channel_config(&self) -> AudioNodeChannelConfig {
        AudioNodeChannelConfig {
            num_inputs: 2,
            num_outputs: 2,
        }
    }

    /// Activate an instance of this node with the the given parameters.
    fn activate(
        &mut self,
        instance_id: u32,
        info: ActiveServerInfo,
        initial_params: &[(ParamID, ParamData)],
    ) -> Box<dyn AudioNodeProcessor>;

    /// Called when the effect becomes inactive (the [`AudioNodeProcessor`] counterpart
    /// has been dropped).
    #[allow(unused)]
    fn on_deactivated(&mut self, instance_id: u32) {}

    /// Returns the display string for the given normalized value for the parameter.
    ///
    /// (i.e. "100%", "-6dB", "12kHz")
    #[allow(unused)]
    fn param_normal_to_string(param_id: u32, normal: NormalVal) -> Result<String, ()> {
        Err(())
    }
}

pub struct AudioNodeChannelConfig {
    pub num_inputs: u16,
    pub num_outputs: u16,
}

/// The real-time processor counterpart to a [`AudioNode`]
pub trait AudioNodeProcessor: Send + 'static {
    /// Process the given buffers.
    fn process(
        &mut self,
        cx: &ProcessContext,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        param_updates: &[ParamUpdate],
        inputs: &[&BlockBuffer<MAX_BLOCK_SIZE>],
        outputs: &mut [&mut BlockBuffer<MAX_BLOCK_SIZE>],
    );
}

#[derive(Debug, Clone, Copy)]
pub struct ParamUpdate {
    /// The unique identifier of the parameter
    pub id: ParamID,

    /// The new data
    pub data: ParamData,

    /// The frame where this parameter update occurs (relative to
    /// the start of the block).
    pub frame_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioResourceKey {
    // TODO
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamData {
    Normal(NormalVal),
    Stepped(u32),
    Boolean(bool),
    Resource(AudioResourceKey),
}

/// Information about a parameter
pub struct ParamInfo {
    /// The unique identifier of this parameter.
    pub id: ParamID,
    /// The display name of this parameter.
    pub name: String,
    /// The number of discrete steps in this parameter.
    ///
    /// If this is `None`, then this parameter is continuous.
    pub num_steps: Option<u32>,
    /// The unit of this parameter.
    pub unit: ParamUnit,
    /// The default normalized value of this parameter.
    pub default: NormalVal,
}

/// The type of audio parameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamType {
    /// A continuous parameter normalized to the range `[0.0, 1.0]`
    Normal {
        /// The unit of this parameter
        unit: ParamUnit,
        /// The default normalized value of this parameter
        default: NormalVal,
    },
    /// A stepped (discrete) parameter
    Stepped {
        /// The number of discrete steps
        num_steps: NonZeroU32,
        /// The default value of this parameter
        default: u32,
    },
    Resource(AudioResouceType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioResouceType {
    Audio,
    AudioStream,
    MIDI,
}

/// The unit of an audio paramter
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamUnit {
    #[default]
    /// A generic unit type
    Generic,
    /// A unit in decibels
    Decibels,
    /// A unit in Hz (cyles per second)
    FreqHz,
}

/// Information about an active audio server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveServerInfo {
    /// The sample rate of the stream
    pub sample_rate: u32,
    /// The maximum block size (the maximum length of each channel slice that will be passed
    /// to [`AudioNodeProcessor::process`])
    pub max_block_size: usize,
    /// The number of input channels
    pub num_in_channels: usize,
    /// The number of output channels
    pub num_out_channels: NonZeroUsize,
}

/// Contains information about the process and loaded resources.
#[derive(Debug, Clone, Copy)]
pub struct ProcessContext {
    /// The instant the output data callback was invoked
    pub callback_timestamp: Instant,

    /// The estimated time between [`StreamCallbackInfo::callback_timestamp`] and the
    /// instant the data will be delivered to the playback device
    pub output_latency: Duration,
    // TODO: include a way to get loaded resources (hashmap maybe?)\

    // Provide sample rate fields for convenience.
    pub sample_rate: f32,
    pub sample_rate_recip: f32,
    pub sample_rate_f64: f64,
    pub sample_rate_recip_f64: f64,
}

impl ProcessContext {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            callback_timestamp: Instant::now(),
            output_latency: Duration::default(),
            sample_rate_f64: sample_rate,
            sample_rate_recip_f64: sample_rate.recip(),
            sample_rate: sample_rate as f32,
            sample_rate_recip: sample_rate.recip() as f32,
        }
    }
}
