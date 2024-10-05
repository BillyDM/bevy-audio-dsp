use std::cell::{Ref, RefCell, RefMut};

use thunderdome::Arena;

use crate::{
    node::{AudioNodeProcessor, ProcInfo, ProcessStatus},
    SilenceMask, MAX_BLOCK_FRAMES,
};

use super::{compiler::CompiledSchedule, NodeID};

pub struct AudioGraphExecutor {
    nodes: Arena<Box<dyn AudioNodeProcessor>>,
    schedule_data: Option<ScheduleHeapData>,

    // TODO: Do research on whether `rtrb` is compatible with
    // webassembly. If not, use conditional compilation to
    // use a different channel type when targeting webassembly.
    from_graph_rx: rtrb::Consumer<GraphToExecutorMsg>,
    to_graph_tx: rtrb::Producer<ExecutorToGraphMsg>,

    num_stream_in_channels: usize,
    num_stream_out_channels: usize,

    stream_in_buffer_list: Option<Vec<RefMut<'static, [f32; MAX_BLOCK_FRAMES]>>>,
    stream_out_buffer_list: Option<Vec<Ref<'static, [f32; MAX_BLOCK_FRAMES]>>>,

    running: bool,
}

impl AudioGraphExecutor {
    pub(crate) fn new(
        from_graph_rx: rtrb::Consumer<GraphToExecutorMsg>,
        to_graph_tx: rtrb::Producer<ExecutorToGraphMsg>,
        max_node_capacity: usize,
        num_stream_in_channels: u16,
        num_stream_out_channels: u16,
    ) -> Self {
        Self {
            nodes: Arena::with_capacity(max_node_capacity),
            schedule_data: None,
            from_graph_rx,
            to_graph_tx,
            num_stream_in_channels: num_stream_in_channels as usize,
            num_stream_out_channels: num_stream_out_channels as usize,
            stream_in_buffer_list: Some(Vec::with_capacity(num_stream_in_channels as usize)),
            stream_out_buffer_list: Some(Vec::with_capacity(num_stream_out_channels as usize)),
            running: true,
        }
    }

    pub fn process_interleaved(&mut self, input: &[f32], output: &mut [f32], frames: usize) {
        if self.schedule_data.is_none() || frames == 0 || !self.running {
            output.fill(0.0);
            return;
        };

        assert_eq!(input.len(), frames * self.num_stream_in_channels);
        assert_eq!(output.len(), frames * self.num_stream_out_channels);

        let mut frames_processed = 0;
        while frames_processed < frames {
            let block_frames = (frames - frames_processed).min(MAX_BLOCK_FRAMES);

            // Prepare graph input buffers.
            {
                // This trick allows us to create a Vec of references without
                // allocating any memory.
                let mut stream_in_buffer_list: Vec<RefMut<[f32; MAX_BLOCK_FRAMES]>> =
                    crate::util::recycle_vec(self.stream_in_buffer_list.take().unwrap());

                let schedule_data = self.schedule_data.as_ref().unwrap();

                let graph_in_buffers = &schedule_data.schedule.schedule
                    [schedule_data.schedule.graph_in_idx]
                    .output_buffers;

                for i in 0..self.num_stream_in_channels {
                    stream_in_buffer_list.push(RefCell::borrow_mut(
                        &schedule_data.buffers[graph_in_buffers[i].buffer_index.0 as usize].0,
                    ));
                }

                if graph_in_buffers.len() > self.num_stream_in_channels {
                    for i in self.num_stream_in_channels..graph_in_buffers.len() {
                        RefCell::borrow_mut(
                            &schedule_data.buffers[graph_in_buffers[i].buffer_index.0 as usize].0,
                        )[0..frames]
                            .fill(0.0);
                    }
                }

                crate::util::deinterleave(
                    &input[frames_processed * self.num_stream_in_channels
                        ..(frames_processed + block_frames) * self.num_stream_in_channels],
                    &mut stream_in_buffer_list,
                );

                self.stream_in_buffer_list = Some(crate::util::recycle_vec(stream_in_buffer_list));
            }

            self.process_block(block_frames);

            // Copy the output of the graph to the output buffer.
            {
                let mut stream_out_buffer_list: Vec<Ref<[f32; MAX_BLOCK_FRAMES]>> =
                    crate::util::recycle_vec(self.stream_out_buffer_list.take().unwrap());

                let schedule_data = self.schedule_data.as_ref().unwrap();

                let graph_out_buffers = &schedule_data.schedule.schedule
                    [schedule_data.schedule.graph_out_idx]
                    .input_buffers;

                for i in 0..self.num_stream_out_channels {
                    stream_out_buffer_list.push(RefCell::borrow(
                        &schedule_data.buffers[graph_out_buffers[i].buffer_index.0 as usize].0,
                    ));
                }

                crate::util::interleave(
                    &stream_out_buffer_list,
                    &mut output[frames_processed * self.num_stream_out_channels
                        ..(frames_processed + block_frames) * self.num_stream_out_channels],
                );

                self.stream_out_buffer_list =
                    Some(crate::util::recycle_vec(stream_out_buffer_list));
            }

            if !self.running {
                if frames_processed < frames {
                    output[frames_processed * self.num_stream_out_channels..].fill(0.0);
                }
                break;
            }

            frames_processed += block_frames;
        }

        if !self.running {
            self.to_graph_tx.push(ExecutorToGraphMsg::Stopped).unwrap();
        }
    }

    fn process_block(&mut self, block_frames: usize) {
        while let Ok(msg) = self.from_graph_rx.pop() {
            match msg {
                GraphToExecutorMsg::NewSchedule(mut new_schedule_data) => {
                    if let Some(mut old_schedule_data) = self.schedule_data.take() {
                        std::mem::swap(
                            &mut old_schedule_data.removed_node_processors,
                            &mut new_schedule_data.removed_node_processors,
                        );

                        for node_id in new_schedule_data.nodes_to_remove.iter() {
                            if let Some(processor) = self.nodes.remove(node_id.0) {
                                old_schedule_data
                                    .removed_node_processors
                                    .push((*node_id, processor));
                            }
                        }

                        self.to_graph_tx
                            .push(ExecutorToGraphMsg::ReturnSchedule(old_schedule_data))
                            .unwrap();
                    }

                    for (node_id, processor) in new_schedule_data.nodes_to_add.drain(..) {
                        assert!(self.nodes.insert_at(node_id.0, processor).is_none());
                    }

                    self.schedule_data = Some(new_schedule_data);
                }
                GraphToExecutorMsg::Stop => {
                    self.running = false;
                }
            }
        }

        if !self.running {
            return;
        }

        let Some(schedule_data) = &mut self.schedule_data else {
            return;
        };
        let ScheduleHeapData {
            schedule,
            buffers,
            in_buffer_list,
            out_buffer_list,
            nodes_to_add: _,
            nodes_to_remove: _,
            removed_node_processors: _,
        } = schedule_data;

        for scheduled_node in schedule.schedule.iter() {
            let mut in_silence_mask = SilenceMask::NONE_SILENT;
            let mut out_silence_mask = SilenceMask::NONE_SILENT;

            {
                // This trick allows us to create a Vec of references without
                // allocating any memory.
                let mut inputs: Vec<Ref<[f32; MAX_BLOCK_FRAMES]>> =
                    crate::util::recycle_vec(in_buffer_list.take().unwrap());
                let mut outputs: Vec<RefMut<[f32; MAX_BLOCK_FRAMES]>> =
                    crate::util::recycle_vec(out_buffer_list.take().unwrap());

                for (i, b) in scheduled_node.input_buffers.iter().enumerate() {
                    if b.should_clear {
                        RefCell::borrow_mut(&buffers[b.buffer_index.0 as usize].0)[0..block_frames]
                            .fill(0.0);

                        in_silence_mask = in_silence_mask.set_channel(i, true);
                    } else if buffers[b.buffer_index.0 as usize].1 {
                        in_silence_mask = in_silence_mask.set_channel(i, true);
                    }

                    inputs.push(RefCell::borrow(&buffers[b.buffer_index.0 as usize].0));
                }

                for b in scheduled_node.output_buffers.iter() {
                    outputs.push(RefCell::borrow_mut(&buffers[b.buffer_index.0 as usize].0));
                }

                let proc_info = ProcInfo {
                    in_silence_mask,
                    out_silence_mask: &mut out_silence_mask,
                };

                let status = self.nodes[scheduled_node.id.0].process(
                    block_frames,
                    proc_info,
                    &inputs,
                    &mut outputs,
                );

                if let ProcessStatus::Err { msg } = status {
                    // TODO: Handle error
                }

                *in_buffer_list = Some(crate::util::recycle_vec(inputs));
                *out_buffer_list = Some(crate::util::recycle_vec(outputs));
            }

            for (i, b) in scheduled_node.output_buffers.iter().enumerate() {
                buffers[b.buffer_index.0 as usize].1 = out_silence_mask.is_channel_silent(i);
            }
        }
    }
}

impl Drop for AudioGraphExecutor {
    fn drop(&mut self) {
        // Make sure the nodes are not deallocated in the audio thread.
        let mut nodes = Arena::new();
        std::mem::swap(&mut nodes, &mut self.nodes);

        let _ = self.to_graph_tx.push(ExecutorToGraphMsg::Dropped {
            nodes,
            schedule_data: self.schedule_data.take(),
        });
    }
}

pub(crate) struct ScheduleHeapData {
    schedule: CompiledSchedule,
    buffers: Vec<(RefCell<[f32; MAX_BLOCK_FRAMES]>, bool)>,
    nodes_to_add: Vec<(NodeID, Box<dyn AudioNodeProcessor>)>,
    nodes_to_remove: Vec<NodeID>,
    removed_node_processors: Vec<(NodeID, Box<dyn AudioNodeProcessor>)>,
    in_buffer_list: Option<Vec<Ref<'static, [f32; MAX_BLOCK_FRAMES]>>>,
    out_buffer_list: Option<Vec<RefMut<'static, [f32; MAX_BLOCK_FRAMES]>>>,
}

pub(crate) enum GraphToExecutorMsg {
    NewSchedule(ScheduleHeapData),
    Stop,
}

pub(crate) enum ExecutorToGraphMsg {
    ReturnSchedule(ScheduleHeapData),
    Stopped,
    Dropped {
        nodes: Arena<Box<dyn AudioNodeProcessor>>,
        schedule_data: Option<ScheduleHeapData>,
    },
}
