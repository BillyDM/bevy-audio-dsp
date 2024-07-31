use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::{
    node::{AudioNodeProcessor, ParamUpdate},
    BlockFrames, EngineToAudioParamUpdate, MAX_BLOCK_SIZE,
};

const PARAM_UPDATE_BUFFER_SIZE: usize = 2048;

pub(crate) struct NodeProcessorState {
    pub node: Box<dyn AudioNodeProcessor>,
    pub param_updates: Vec<ParamUpdate>,
    pub param_update_offsets: Vec<usize>,

    engine_param_updates: VecDeque<EngineToAudioParamUpdate>,
}

impl NodeProcessorState {
    fn new(node: Box<dyn AudioNodeProcessor>) -> Self {
        Self {
            node,
            param_updates: Vec::with_capacity(PARAM_UPDATE_BUFFER_SIZE),
            param_update_offsets: Vec::with_capacity(PARAM_UPDATE_BUFFER_SIZE),
            engine_param_updates: VecDeque::with_capacity(PARAM_UPDATE_BUFFER_SIZE),
        }
    }

    pub fn push_engine_param_update(&mut self, update: EngineToAudioParamUpdate) {
        if self.engine_param_updates.is_empty() {
            self.engine_param_updates.push_back(update);
            return;
        }

        let last_instant = self.engine_param_updates.back().unwrap().instant;

        // If the update happens on or after the last update in the queue,
        // simply push it on the end (this is the likely case).
        if update.instant >= last_instant {
            self.engine_param_updates.push_back(update);
            return;
        }

        // Otherwise, insert the update in the correct place (this is the
        // unlikely case).
        let mut insert_i = 0;
        for (i, u) in self.engine_param_updates.iter().enumerate() {
            if update.instant < u.instant {
                insert_i = i;
                break;
            }
        }
        self.engine_param_updates.insert(insert_i, update);
    }

    fn prepare_param_updates(
        &mut self,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        block_start_instant: Instant,
        block_end_instant: Instant,
        sample_rate: f64,
    ) {
        self.param_updates.clear();
        self.param_update_offsets.clear();

        loop {
            let Some(update_instant) = self.engine_param_updates.front().map(|u| u.instant) else {
                break;
            };

            if update_instant >= block_end_instant {
                break;
            }

            let update = self.engine_param_updates.pop_front().unwrap();

            let frame_offset = if update_instant <= block_start_instant {
                0
            } else {
                let frame_offset = (update_instant
                    .duration_since(block_start_instant)
                    .as_secs_f64()
                    * sample_rate)
                    .round() as usize;

                if frame_offset >= frames.get() {
                    frames.get() - 1
                } else {
                    frame_offset
                }
            };

            self.param_updates.push(ParamUpdate {
                id: update.param_id,
                data: update.data,
                frame_offset: frame_offset as u32,
            });
        }
    }
}

pub(crate) struct NodeProcessorPool {
    pub nodes: Vec<NodeProcessorState>,
}

impl NodeProcessorPool {
    pub fn new(nodes: Vec<Box<dyn AudioNodeProcessor>>) -> Self {
        Self {
            nodes: nodes
                .into_iter()
                .map(|n| NodeProcessorState::new(n))
                .collect(),
        }
    }

    pub fn push_engine_param_update(&mut self, update: EngineToAudioParamUpdate) {
        self.nodes[update.node_id].push_engine_param_update(update);
    }

    pub fn prepare_param_updates(
        &mut self,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        block_start_instant: Instant,
        sample_rate: f64,
        sample_rate_recip: f64,
    ) {
        let block_end_instant =
            block_start_instant + Duration::from_secs_f64(frames.get() as f64 * sample_rate_recip);

        for node in self.nodes.iter_mut() {
            node.prepare_param_updates(frames, block_start_instant, block_end_instant, sample_rate);
        }
    }
}
