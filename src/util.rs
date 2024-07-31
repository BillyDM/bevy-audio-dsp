//! General conversion functions and utilities.

use std::cell::{Ref, RefMut};

use crate::{
    graph::buffer_pool::{BlockBuffer, BlockRange},
    node::ParamUpdate,
    BlockFrames, MAX_BLOCK_SIZE,
};

/// Returns the raw amplitude from the given decibel value.
#[inline]
pub fn db_to_amp(db: f32) -> f32 {
    10.0f32.powf(0.05 * db)
}

/// Returns the decibel value from the raw amplitude.
#[inline]
pub fn amp_to_db(amp: f32) -> f32 {
    20.0 * amp.log(10.0)
}

/// Returns the raw amplitude from the given decibel value.
///
/// If `db <= -100.0`, then 0.0 will be returned instead (negative infinity gain).
#[inline]
pub fn db_to_amp_clamped_neg_100_db(db: f32) -> f32 {
    if db <= -100.0 {
        0.0
    } else {
        db_to_amp(db)
    }
}

/// Returns the decibel value from the raw amplitude value.
///
/// If `amp <= 0.00001`, then the minimum of `-100.0` dB will be
/// returned instead (representing negative infinity gain when paired with
/// [`db_to_amp_clamped_neg_100_db`]).
#[inline]
pub fn amp_to_db_clamped_neg_100_db(amp: f32) -> f32 {
    if amp <= 0.00001 {
        -100.0
    } else {
        amp_to_db(amp)
    }
}

/// Efficiently deinterleave audio.
pub fn deinterleave<const MAX_BLOCK_SIZE: usize>(
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    interleaved: &[f32],
    channels: &mut [RefMut<'_, BlockBuffer<MAX_BLOCK_SIZE>>],
) {
    let frames = frames.get() as usize;

    match channels.len() {
        0 => return,
        1 => {
            let min_len = frames.min(interleaved.len());
            channels[0].data[0..interleaved.len()].copy_from_slice(&interleaved[0..min_len]);
        }
        // Provide a loop with optimized stereo deinterleaving
        2 => {
            let (ch0, ch1) = channels.split_first_mut().unwrap();
            let ch0 = &mut ch0.data[0..frames];
            let ch1 = &mut ch1[0].data[0..frames];

            for (input, (ch0s, ch1s)) in interleaved
                .chunks_exact(2)
                .zip(ch0.iter_mut().zip(ch1.iter_mut()))
            {
                *ch0s = input[0];
                *ch1s = input[1];
            }
        }
        n => {
            for (ch_i, ch) in channels.iter_mut().enumerate() {
                for (input, output) in interleaved
                    .iter()
                    .skip(ch_i)
                    .step_by(n)
                    .zip(ch.data.iter_mut())
                {
                    *output = *input;
                }
            }
        }
    }
}

/// Efficiently interleave audio.
pub fn interleave<const MAX_BLOCK_SIZE: usize>(
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    channels: &[Ref<'_, BlockBuffer<MAX_BLOCK_SIZE>>],
    interleaved: &mut [f32],
) {
    let frames = frames.get() as usize;

    match channels.len() {
        0 => return,
        1 => {
            if channels[0].is_silent {
                return;
            }

            let min_len = frames.min(interleaved.len());
            interleaved[0..min_len].copy_from_slice(&channels[0].data[0..min_len]);
        }
        // Provide a loop with optimized stereo interleaving
        2 => {
            if channels[0].is_silent && channels[1].is_silent {
                return;
            }

            let ch0 = &channels[0].data[0..frames];
            let ch1 = &channels[1].data[0..frames];

            for (output, (ch0s, ch1s)) in interleaved
                .chunks_exact_mut(2)
                .zip(ch0.iter().zip(ch1.iter()))
            {
                output[0] = *ch0s;
                output[1] = *ch1s;
            }
        }
        n => {
            for (ch_i, ch) in channels.iter().enumerate() {
                if ch.is_silent {
                    continue;
                }

                for (output, input) in interleaved
                    .iter_mut()
                    .skip(ch_i)
                    .step_by(n)
                    .zip(ch.data.iter())
                {
                    *output = *input;
                }
            }
        }
    }
}

/// Convenience function to mutably borrow two output buffers at the
/// same time.
#[inline]
pub fn output_stereo<'a>(
    outputs: &'a mut [&mut BlockBuffer<MAX_BLOCK_SIZE>],
) -> (
    &'a mut BlockBuffer<MAX_BLOCK_SIZE>,
    &'a mut BlockBuffer<MAX_BLOCK_SIZE>,
) {
    let (l, r) = outputs.split_first_mut().unwrap();
    (l, &mut r[0])
}

/// Process in chunks, where each new chunk occurs at each new chronological
/// parameter update.
pub fn param_update_chunks<F: FnMut(&[ParamUpdate], BlockRange<MAX_BLOCK_SIZE>)>(
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    param_updates: &[ParamUpdate],
    mut f: F,
) {
    let frames = frames.get() as u32;

    let mut frames_processed: u32 = 0;
    let mut param_updates_processed = 0;
    while frames_processed < frames {
        let mut num_chunk_param_updates = 0;
        let mut chunk_frames = frames - frames_processed;

        for update in param_updates.iter().skip(param_updates_processed) {
            if update.frame_offset > frames_processed {
                chunk_frames = (update.frame_offset - frames_processed).min(chunk_frames);
                break;
            } else {
                num_chunk_param_updates += 1;
            }
        }

        #[cfg(debug_assertions)]
        let (param_updates, range) = {
            (
                &param_updates
                    [param_updates_processed..param_updates_processed + num_chunk_param_updates],
                BlockRange::new(frames_processed..frames_processed + chunk_frames).unwrap(),
            )
        };

        #[cfg(not(debug_assertions))]
        let (param_updates, range) = {
            // # SAFETY:
            // The way the logic of this loop is set up, these values cannot be out of bounds.
            // The compiler just isn't smart enough to reason about this.
            unsafe {
                (
                    std::slice::from_raw_parts(
                        param_updates.as_ptr().add(param_updates_processed),
                        num_chunk_param_updates,
                    ),
                    BlockRange::new_unchecked(frames_processed..frames_processed + chunk_frames),
                )
            }
        };

        (f)(param_updates, range);

        frames_processed += chunk_frames;
        param_updates_processed += num_chunk_param_updates;
    }
}
