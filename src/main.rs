use std::{
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

mod util;

const MAX_BLOCK_SIZE: usize = 256;
const GUI_TO_AUDIO_MSG_SIZE: usize = 2048;

fn main() {
    env_logger::init();

    let mut audio = AudioResource::new();

    let _cpal_out_stream = {
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();
        let config = device.default_output_config().unwrap();

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        assert_eq!(channels, 2);

        let error_callback = |err| eprintln!("an error occurred on output stream: {}", err);

        let mut audio_processor = audio.activate(ActiveServerInfo {
            sample_rate,
            max_block_size: MAX_BLOCK_SIZE,
            num_in_channels: 0,
            num_out_channels: NonZeroUsize::new(channels).unwrap(),
        });

        let sample_rate_recip = (sample_rate as f64).recip();

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

enum GuiToAudioMsg {}

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
    to_audio_tx: rtrb::Producer<GuiToAudioMsg>,
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
    pub fn activate(&mut self, info: ActiveServerInfo) -> AudioProcessor {
        let (to_audio_tx, from_gui_rx) =
            rtrb::RingBuffer::<GuiToAudioMsg>::new(GUI_TO_AUDIO_MSG_SIZE);

        self.active_state = Some(ActiveServerState { to_audio_tx, info });

        AudioProcessor {
            from_gui_rx,
            max_block_size: info.max_block_size,
            input_channels: vec![vec![0.0; info.max_block_size]; info.num_in_channels],
            output_channels: vec![vec![0.0; info.max_block_size]; info.num_out_channels.into()],
            phasor: 0.0,
            phasor_inc: 440.0 / info.sample_rate as f32,
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
    /// The maximum block size
    pub max_block_size: usize,
    /// The number of input channels
    pub num_in_channels: usize,
    /// The number of output channels
    pub num_out_channels: NonZeroUsize,
}

/// The audio-thread part of the audio server.
pub struct AudioProcessor {
    from_gui_rx: rtrb::Consumer<GuiToAudioMsg>,

    max_block_size: usize,

    input_channels: Vec<Vec<f32>>,
    output_channels: Vec<Vec<f32>>,

    phasor: f32,
    phasor_inc: f32,
}

impl AudioProcessor {
    /// Process the given buffers.
    pub fn process_interleaved(
        &mut self,
        input_buffer: &[f32],
        output_buffer: &mut [f32],
        info: &StreamCallbackInfo,
    ) {
        let frames = output_buffer.len() / self.output_channels.len();

        // Process in blocks
        let mut frames_processed = 0;
        while frames_processed < frames {
            let block_frames = (frames - frames_processed).min(self.max_block_size);

            if !self.input_channels.is_empty() {
                crate::util::deinterleave(
                    &input_buffer[frames_processed * self.input_channels.len()
                        ..(frames_processed + block_frames) * self.input_channels.len()],
                    &mut self.input_channels,
                );
            }

            self.process_block(block_frames, info);

            crate::util::interleave(
                &self.output_channels,
                &mut output_buffer[frames_processed * self.output_channels.len()
                    ..(frames_processed + block_frames) * self.output_channels.len()],
            );

            frames_processed += block_frames;
        }
    }

    fn process_block(&mut self, frames: usize, _info: &StreamCallbackInfo) {
        let input_silence_mask = SilenceMask::from_channels_slow(&self.input_channels, frames);

        for b in self.output_channels.iter_mut() {
            b[0..frames].fill(0.0);
        }
        let mut output_silence_mask = SilenceMask::new_all_silent(self.output_channels.len());

        let (out_l, out_r) = self.output_channels.split_first_mut().unwrap();
        let out_l = &mut out_l[0..frames];
        let out_r = &mut out_r.first_mut().unwrap()[0..frames];

        for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
            // Generate a sine wave at 440 Hz at 25% volume.
            let value = (self.phasor * std::f32::consts::TAU).sin() * 0.25;
            self.phasor = (self.phasor + self.phasor_inc).fract();

            *l = value;
            *r = value;
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

/// A mask which specifies which channels contain silence.
///
/// This can be used for optimization by skipping processing for inputs
/// that contain silence.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilenceMask(pub u32);

impl SilenceMask {
    /// Create a new silence mask from the channels by thouroughly
    /// checking every sample.
    pub fn from_channels_slow(channels: &[Vec<f32>], frames: usize) -> Self {
        if !channels.is_empty() {
            let mut mask: u32 = 0;

            for (ch_i, ch) in channels.iter().enumerate() {
                let mut is_silent = true;

                for val in &ch[0..frames] {
                    if *val != 0.0 {
                        is_silent = false;
                        break;
                    }
                }

                if is_silent {
                    mask |= 1 << ch_i;
                }
            }

            Self(mask)
        } else {
            Self(0)
        }
    }

    /// Create a new silence mask with all flags set.
    pub fn new_all_silent(num_channels: usize) -> Self {
        if num_channels == 0 {
            Self(0)
        } else {
            Self((1 << num_channels) - 1)
        }
    }

    /// Returns whether or not all flags are set for all channels.
    pub fn all_channels_silent_fast(&self, channels: &[Vec<f32>]) -> bool {
        if channels.is_empty() {
            true
        } else {
            let num_channels = channels.len();
            let all_silent_mask = (1 << num_channels) - 1;
            self.0 & all_silent_mask == all_silent_mask
        }
    }

    /// Returns whether or not the silent flag is set for a given channel.
    #[inline]
    pub fn is_channel_silent_fast(&self, index: usize) -> bool {
        self.0 & (1 << index) != 0
    }
}
