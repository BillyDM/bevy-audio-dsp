use std::{
    cell::{Ref, RefMut},
    ops::{Deref, DerefMut},
};

use arrayvec::ArrayVec;

use crate::{
    node::{ProcessContext, MAX_IN_CHANNELS, MAX_OUT_CHANNELS},
    BlockFrames, MAX_BLOCK_SIZE,
};

use super::{
    buffer_pool::{BlockBuffer, BufferPool},
    node_pool::NodeProcessorPool,
};

pub(crate) type BufferId = usize;
pub(crate) type NodeID = usize;

pub(crate) struct Schedule {
    pub tasks: Vec<ProcessTask>,
    pub num_buffers: usize,
    pub input_buffer_ids: ArrayVec<BufferId, MAX_IN_CHANNELS>,
    pub output_buffer_ids: ArrayVec<BufferId, MAX_OUT_CHANNELS>,
}

impl Schedule {
    pub(crate) fn process_block_interleaved(
        &mut self,
        cx: &ProcessContext,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
        nodes: &mut NodeProcessorPool,
        input_interleaved: &[f32],
        output_interleaved: &mut [f32],
    ) {
        assert_eq!(
            input_interleaved.len(),
            frames.get() * self.input_buffer_ids.len()
        );
        assert_eq!(
            output_interleaved.len(),
            frames.get() * self.output_buffer_ids.len()
        );

        // Prepare input buffers
        {
            let mut in_channels_refs: ArrayVec<
                RefMut<'_, BlockBuffer<MAX_BLOCK_SIZE>>,
                MAX_IN_CHANNELS,
            > = self
                .input_buffer_ids
                .iter()
                .map(|id| buffers.buffer_mut(*id))
                .collect();

            crate::util::deinterleave(frames, input_interleaved, in_channels_refs.as_mut_slice());

            for ch in in_channels_refs.iter_mut() {
                let mut is_silent = true;
                for s in ch.data[0..frames.get()].iter() {
                    if *s != 0.0 {
                        is_silent = false;
                        break;
                    }
                }

                ch.is_silent = is_silent;
            }
        }

        self.process_block(cx, frames, buffers, nodes);

        // Extract output buffers
        {
            let out_channels_refs: ArrayVec<
                Ref<'_, BlockBuffer<MAX_BLOCK_SIZE>>,
                MAX_OUT_CHANNELS,
            > = self
                .output_buffer_ids
                .iter()
                .map(|id| buffers.buffer(*id))
                .collect();

            crate::util::interleave(frames, out_channels_refs.as_slice(), output_interleaved);
        }
    }

    fn process_block(
        &mut self,
        cx: &ProcessContext,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
        nodes: &mut NodeProcessorPool,
    ) {
        for task in self.tasks.iter_mut() {
            task.process(cx, frames, buffers, nodes);
        }
    }
}

pub enum ProcessTask {
    Node {
        node_id: NodeID,
        input_ids: ArrayVec<BufferId, MAX_IN_CHANNELS>,
        output_ids: ArrayVec<BufferId, MAX_OUT_CHANNELS>,
    },
    Sum {
        input_ids: Vec<BufferId>,
        output_id: BufferId,
    },
    Clear(Vec<BufferId>),
}

impl ProcessTask {
    fn process(
        &self,
        cx: &ProcessContext,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
        nodes: &mut NodeProcessorPool,
    ) {
        match self {
            ProcessTask::Sum {
                input_ids,
                output_id,
            } => {
                sum(frames, input_ids, *output_id, buffers);
            }
            ProcessTask::Clear(buffer_ids) => {
                clear(frames, buffer_ids, buffers);
            }
            ProcessTask::Node {
                node_id,
                input_ids,
                output_ids,
            } => {
                process_node(cx, frames, *node_id, input_ids, output_ids, buffers, nodes);
            }
        }
    }
}

fn clear<const MAX_BLOCK_SIZE: usize>(
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    buffer_ids: &[BufferId],
    buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
) {
    for id in buffer_ids.iter().copied() {
        let mut buf = buffers.buffer_mut(id);
        buf.data[0..frames.get()].fill(0.0);
        buf.is_silent = true;
    }
}

fn sum<const MAX_BLOCK_SIZE: usize>(
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    input_ids: &[BufferId],
    output_id: BufferId,
    buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
) {
    let mut output = buffers.buffer_mut(output_id);

    for id in input_ids.iter().copied() {
        let input = buffers.buffer(id);

        if !input.is_silent {
            for i in 0..frames.get() {
                output.data[i] += input.data[i];
            }
            output.is_silent = false;
        }
    }
}

fn process_node(
    cx: &ProcessContext,
    frames: BlockFrames<MAX_BLOCK_SIZE>,
    node_id: NodeID,
    input_ids: &ArrayVec<BufferId, MAX_IN_CHANNELS>,
    output_ids: &ArrayVec<BufferId, MAX_OUT_CHANNELS>,
    buffers: &mut BufferPool<MAX_BLOCK_SIZE>,
    nodes: &mut NodeProcessorPool,
) {
    let in_buffer_refs: ArrayVec<Ref<'_, BlockBuffer<MAX_BLOCK_SIZE>>, MAX_IN_CHANNELS> =
        input_ids.iter().map(|id| buffers.buffer(*id)).collect();
    let mut out_buffer_refs: ArrayVec<RefMut<'_, BlockBuffer<MAX_BLOCK_SIZE>>, MAX_OUT_CHANNELS> =
        output_ids
            .iter()
            .map(|id| {
                let mut buf = buffers.buffer_mut(*id);
                buf.is_silent = false;
                buf
            })
            .collect();

    let inputs: ArrayVec<&BlockBuffer<MAX_BLOCK_SIZE>, MAX_IN_CHANNELS> =
        in_buffer_refs.iter().map(|i| i.deref()).collect();
    let mut outputs: ArrayVec<&mut BlockBuffer<MAX_BLOCK_SIZE>, MAX_OUT_CHANNELS> =
        out_buffer_refs.iter_mut().map(|i| i.deref_mut()).collect();

    let node_state = &mut nodes.nodes[node_id];

    node_state.node.process(
        cx,
        frames,
        &node_state.param_updates,
        inputs.as_slice(),
        outputs.as_mut_slice(),
    );
}
