use std::{
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use arrayvec::ArrayVec;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

mod built_in_nodes;
mod graph;

pub mod node;
pub mod util;

pub use graph::buffer_pool::{BlockBuffer, BlockFrames, BlockRange};
pub use node::{MAX_IN_CHANNELS, MAX_OUT_CHANNELS};

use graph::{
    buffer_pool::BufferPool,
    node_pool::NodeProcessorPool,
    schedule::{NodeID, Schedule},
    CompiledAudioGraph,
};
use node::ProcessContext;

pub const MAX_BLOCK_SIZE: usize = 128;
const GUI_TO_AUDIO_MSG_SIZE: usize = 2048;

fn main() {
    env_logger::init();

    let mut audio = AudioResource::new();

    let nodes = vec![];

    let mut output_buffer_ids = ArrayVec::new();
    output_buffer_ids.push(0);
    output_buffer_ids.push(1);

    let schedule = Schedule {
        tasks: Vec::new(),
        num_buffers: 2,
        input_buffer_ids: ArrayVec::new(),
        output_buffer_ids,
    };

    let compiled_graph = CompiledAudioGraph { schedule, nodes };

    let _cpal_out_stream = {
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();
        let config = device.default_output_config().unwrap();

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        assert_eq!(channels, 2);

        let error_callback = |err| eprintln!("an error occurred on output stream: {}", err);

        let mut audio_processor = audio.activate(
            ActiveServerInfo {
                sample_rate,
                num_in_channels: 0,
                num_out_channels: NonZeroUsize::new(channels).unwrap(),
            },
            compiled_graph,
        );

        let sample_rate_recip = audio_processor.cx.sample_rate_recip_f64;

        let stream = device
            .build_output_stream(
                &config.into(),
                move |output: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    audio_processor.process_interleaved(
                        &[],
                        output,
                        &StreamCallbackInfo {
                            callback_timestamp: Instant::now(),
                            output_latency: info
                                .timestamp()
                                .playback
                                .duration_since(&info.timestamp().callback)
                                .unwrap_or_else(|| {
                                    Duration::from_secs_f64(
                                        (output.len() / channels) as f64 * sample_rate_recip,
                                    )
                                }),
                        },
                    );
                },
                error_callback,
                None,
            )
            .unwrap();
        stream.play().unwrap();

        stream
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Ok(Box::new(App { audio }))),
    )
    .unwrap();
}

pub struct EngineToAudioParamUpdate {
    /// The id of the node
    pub node_id: NodeID,

    /// The unique identifier of the parameter
    pub param_id: crate::node::ParamID,

    /// The new data
    pub data: crate::node::ParamData,

    /// The instant that this event occurs
    pub instant: Instant,
}

enum EngineToAudioMsg {
    ParamUpdate(EngineToAudioParamUpdate),
}

struct App {
    audio: AudioResource,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");
        });
    }
}

struct ActiveServerState {
    to_audio_tx: rtrb::Producer<EngineToAudioMsg>,
    info: ActiveServerInfo,
}

/// A resource containing the audio server.
pub struct AudioResource {
    active_state: Option<ActiveServerState>,
}

impl AudioResource {
    /// Create a new audio server
    pub fn new() -> Self {
        Self { active_state: None }
    }

    /// Activate the audio server with the given parameters.
    pub fn activate(
        &mut self,
        info: ActiveServerInfo,
        compiled_graph: CompiledAudioGraph,
    ) -> AudioProcessor {
        let (to_audio_tx, from_engine_rx) =
            rtrb::RingBuffer::<EngineToAudioMsg>::new(GUI_TO_AUDIO_MSG_SIZE);

        self.active_state = Some(ActiveServerState { to_audio_tx, info });

        let num_buffers = compiled_graph.schedule.num_buffers;

        AudioProcessor {
            from_engine_rx,
            buffer_pool: BufferPool::new(num_buffers),
            node_pool: NodeProcessorPool::new(compiled_graph.nodes),
            schedule: compiled_graph.schedule,
            cx: ProcessContext::new(info.sample_rate as f64),
        }
    }

    /// Notify the server that the audio processor counterpart has been dropped.
    pub fn on_deactivated(&mut self) {
        self.active_state = None;
    }

    /// Returns whether or not the server is currently active and processing audio.
    pub fn is_active(&self) -> bool {
        self.active_state.is_some()
    }

    /// Returns information about the activated server.
    pub fn active_info(&self) -> Option<&ActiveServerInfo> {
        self.active_state.as_ref().map(|s| &s.info)
    }
}

/// Information about an active audio server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveServerInfo {
    /// The sample rate of the stream
    pub sample_rate: u32,
    /// The number of input channels
    pub num_in_channels: usize,
    /// The number of output channels
    pub num_out_channels: NonZeroUsize,
}

/// The audio-thread part of the audio server.
pub struct AudioProcessor {
    from_engine_rx: rtrb::Consumer<EngineToAudioMsg>,

    buffer_pool: BufferPool<MAX_BLOCK_SIZE>,
    node_pool: NodeProcessorPool,
    schedule: Schedule,

    cx: ProcessContext,
}

impl AudioProcessor {
    /// Process the given buffers.
    pub fn process_interleaved(
        &mut self,
        input_interleaved: &[f32],
        output_interleaved: &mut [f32],
        info: &StreamCallbackInfo,
    ) {
        while let Ok(msg) = self.from_engine_rx.pop() {
            match msg {
                EngineToAudioMsg::ParamUpdate(update) => {
                    self.node_pool.push_engine_param_update(update);
                }
            }
        }

        let frames = output_interleaved.len() / self.schedule.output_buffer_ids.len();

        // Process in blocks
        let mut frames_processed = 0;
        while frames_processed < frames {
            let block_frames = BlockFrames::<MAX_BLOCK_SIZE>::new(
                (frames - frames_processed).min(MAX_BLOCK_SIZE) as u32,
            )
            .unwrap();

            self.cx.callback_timestamp = info.callback_timestamp;
            self.cx.output_latency = info.output_latency;

            self.node_pool.prepare_param_updates(
                block_frames,
                info.callback_timestamp
                    + Duration::from_secs_f64(
                        frames_processed as f64 * self.cx.sample_rate_recip_f64,
                    ),
                self.cx.sample_rate_f64,
                self.cx.sample_rate_recip_f64,
            );

            self.schedule.process_block_interleaved(
                &self.cx,
                block_frames,
                &mut self.buffer_pool,
                &mut self.node_pool,
                &input_interleaved[frames_processed * self.schedule.input_buffer_ids.len()
                    ..(frames_processed + block_frames.get() as usize)
                        * self.schedule.input_buffer_ids.len()],
                &mut output_interleaved[frames_processed * self.schedule.output_buffer_ids.len()
                    ..(frames_processed + block_frames.get() as usize)
                        * self.schedule.output_buffer_ids.len()],
            );

            frames_processed += block_frames.get() as usize;
        }
    }
}

/// Additional info about an audio stream for the [`AudioProcessor::process`] callback.
pub struct StreamCallbackInfo {
    /// The instant the output data callback was invoked.
    pub callback_timestamp: Instant,

    /// The estimated time between [`StreamCallbackInfo::callback_timestamp`] and the
    /// instant the data will be delivered to the playback device.
    pub output_latency: Duration,
    // TODO
    // /// The estimated time between when the data was read from the input device and
    // /// [`StreamCallbackInfo::callback_timestamp`].
    // pub input_latency: Duration,
}

/// A value normalized to the range `[0.0, 1.0]`
#[repr(transparent)]
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NormalVal(f32);

impl NormalVal {
    /// A value of `0.0`
    pub const ZERO: Self = Self(0.0);
    /// A value of `0.5`
    pub const HALF: Self = Self(0.5);
    /// A value of `1.0`
    pub const ONE: Self = Self(1.0);

    /// Construct a new value normalized to the range `[0.0, 1.0]`.
    ///
    /// The value will be clamped to the range `[0.0, 1.0]`.
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the normalized value in the range `[0.0, 1.0]`
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl From<f32> for NormalVal {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<NormalVal> for f32 {
    fn from(value: NormalVal) -> Self {
        value.get()
    }
}
