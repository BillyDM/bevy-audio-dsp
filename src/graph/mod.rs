use schedule::Schedule;

use crate::node::AudioNodeProcessor;

pub mod buffer_pool;
pub mod node_pool;
pub mod schedule;

pub struct CompiledAudioGraph {
    pub(crate) schedule: Schedule,
    pub(crate) nodes: Vec<Box<dyn AudioNodeProcessor>>,
}
