mod compiler;
mod error;
mod executor;

use ahash::AHashSet;
use error::CompileGraphError;
use executor::{ExecutorToGraphMsg, GraphToExecutorMsg};
use thunderdome::Arena;

use compiler::CompiledSchedule;

pub use compiler::{Edge, EdgeID, InPortIdx, NodeEntry, NodeID, OutPortIdx};
pub use error::AddEdgeError;
pub use executor::AudioGraphExecutor;

use crate::node::{AudioNode, DummyAudioNode};

const CHANNEL_CAPACITY: usize = 64;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct EdgeHash {
    pub src_node: NodeID,
    pub src_port: OutPortIdx,
    pub dst_node: NodeID,
    pub dst_port: InPortIdx,
}

struct Channel {
    // TODO: Do research on whether `rtrb` is compatible with
    // webassembly. If not, use conditional compilation to
    // use a different channel type when targeting webassembly.
    to_executor_tx: rtrb::Producer<GraphToExecutorMsg>,
    from_executor_rx: rtrb::Consumer<ExecutorToGraphMsg>,
}

/// A helper struct to construct and modify audio graphs.
pub struct AudioGraph {
    nodes: Arena<NodeEntry<Box<dyn AudioNode>>>,
    edges: Arena<Edge>,
    connected_input_ports: AHashSet<(NodeID, InPortIdx)>,
    existing_edges: AHashSet<EdgeHash>,

    graph_in_id: NodeID,
    graph_out_id: NodeID,

    channel: Option<Channel>,

    needs_compile: bool,
}

impl AudioGraph {
    /// Construct a new [AudioGraph] with some initial allocated capacity.
    fn with_capacity(
        num_graph_inputs: u16,
        num_graph_outputs: u16,
        node_capacity: usize,
        edge_capacity: usize,
    ) -> Self {
        let mut nodes = Arena::<NodeEntry<Box<dyn AudioNode>>>::with_capacity(node_capacity);

        let graph_in_id = NodeID(nodes.insert(NodeEntry::new(
            0,
            num_graph_inputs,
            Box::new(DummyAudioNode),
        )));
        let graph_out_id = NodeID(nodes.insert(NodeEntry::new(
            num_graph_outputs,
            0,
            Box::new(DummyAudioNode),
        )));

        Self {
            nodes,
            edges: Arena::with_capacity(edge_capacity),
            connected_input_ports: AHashSet::with_capacity(edge_capacity),
            existing_edges: AHashSet::with_capacity(edge_capacity),
            graph_in_id,
            graph_out_id,
            channel: None,
            needs_compile: false,
        }
    }

    pub fn create_executor(
        &mut self,
        num_stream_in_channels: u16,
        num_stream_out_channels: u16,
    ) -> Result<AudioGraphExecutor, CompileGraphError> {
        if self.channel.is_some() {
            return Err(CompileGraphError::ExecutorAlreadyExists);
        }

        let (to_executor_tx, from_graph_rx) =
            rtrb::RingBuffer::<GraphToExecutorMsg>::new(CHANNEL_CAPACITY);
        let (to_graph_tx, from_executor_rx) =
            rtrb::RingBuffer::<ExecutorToGraphMsg>::new(CHANNEL_CAPACITY);

        self.channel = Some(Channel {
            to_executor_tx,
            from_executor_rx,
        });

        self.needs_compile = true;
        self.compile_and_send_schedule()?;

        Ok(AudioGraphExecutor::new(
            from_graph_rx,
            to_graph_tx,
            self.nodes.capacity(),
            num_stream_in_channels,
            num_stream_out_channels,
        ))
    }

    pub fn on_stream_closed(&mut self, old_executor: Option<AudioGraphExecutor>) {
        self.channel = None;
        // Todo: cleanup old executor
    }

    /// The ID of the graph input node
    pub fn graph_in_node(&self) -> NodeID {
        self.graph_in_id
    }

    /// The ID of the graph output node
    pub fn graph_out_node(&self) -> NodeID {
        self.graph_out_id
    }

    /// Add a new [Node] the the audio graph.
    ///
    /// This will return the globally unique ID assigned to this node.
    pub fn add_node(&mut self, num_inputs: u16, num_outputs: u16, node: impl AudioNode) -> NodeID {
        self.needs_compile = true;

        let new_id = NodeID(self.nodes.insert(NodeEntry::new(
            num_inputs,
            num_outputs,
            Box::new(node),
        )));
        self.nodes[new_id.0].id = new_id;

        new_id
    }

    /// Get an immutable reference to the node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node(&self, node_id: NodeID) -> Option<&Box<dyn AudioNode>> {
        self.nodes.get(node_id.0).map(|n| &n.weight)
    }

    /// Get a mutable reference to the node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node_mut(&mut self, node_id: NodeID) -> Option<&mut Box<dyn AudioNode>> {
        self.nodes.get_mut(node_id.0).map(|n| &mut n.weight)
    }

    /// Get info about a node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node_info(&self, node_id: NodeID) -> Option<&NodeEntry<Box<dyn AudioNode>>> {
        self.nodes.get(node_id.0)
    }

    /// Remove the given node from the graph.
    ///
    /// This will automatically remove all edges from the graph that
    /// were connected to this node.
    ///
    /// On success, this returns a list of all edges that were removed
    /// from the graph as a result of removing this node.
    ///
    /// This will return an error if a node with the given ID does not
    /// exist in the graph, or if the ID is of the graph input or graph
    /// output node.
    pub fn remove_node(
        &mut self,
        node_id: NodeID,
    ) -> Result<(Box<dyn AudioNode>, Vec<EdgeID>), ()> {
        if node_id == self.graph_in_id || node_id == self.graph_out_id {
            return Err(());
        }

        let node_entry = self.nodes.remove(node_id.0).ok_or(())?;

        let mut removed_edges: Vec<EdgeID> = Vec::new();

        for port_idx in 0..node_entry.num_inputs {
            removed_edges
                .append(&mut self.remove_edges_with_input_port(node_id, InPortIdx(port_idx)));
        }
        for port_idx in 0..node_entry.num_outputs {
            removed_edges
                .append(&mut self.remove_edges_with_output_port(node_id, OutPortIdx(port_idx)));
        }

        for port_idx in 0..node_entry.num_inputs {
            self.connected_input_ports
                .remove(&(node_id, InPortIdx(port_idx)));
        }

        self.needs_compile = true;
        Ok((node_entry.weight, removed_edges))
    }

    /// Get a list of all the existing nodes in the graph.
    pub fn nodes<'a>(&'a self) -> impl Iterator<Item = &'a NodeEntry<Box<dyn AudioNode>>> {
        self.nodes.iter().map(|(_, n)| n)
    }

    /// Get a list of all the existing edges in the graph.
    pub fn edges<'a>(&'a self) -> impl Iterator<Item = &'a Edge> {
        self.edges.iter().map(|(_, e)| e)
    }

    /// Set the number of input ports for a particular node in the graph.
    ///
    /// This will return an error if a node with the given ID does not
    /// exist in the graph, or if the ID is of the graph input node.
    pub fn set_num_inputs(&mut self, node_id: NodeID, num_inputs: u16) -> Result<Vec<EdgeID>, ()> {
        if node_id == self.graph_in_id {
            return Err(());
        }

        let node_entry = self.nodes.get_mut(node_id.0).ok_or(())?;

        let old_num_inputs = node_entry.num_inputs;
        let mut removed_edges = Vec::new();
        if num_inputs < old_num_inputs {
            for port_idx in num_inputs..old_num_inputs {
                removed_edges
                    .append(&mut self.remove_edges_with_input_port(node_id, InPortIdx(port_idx)));
                self.connected_input_ports
                    .remove(&(node_id, InPortIdx(port_idx)));
            }
        }

        self.nodes[node_id.0].num_inputs = num_inputs;

        self.needs_compile = true;
        Ok(removed_edges)
    }

    /// Set the number of output ports for a particular node in the graph.
    ///
    /// This will return an error if a node with the given ID does not
    /// exist in the graph, or if the ID is of the graph output node.
    pub fn set_num_outputs(
        &mut self,
        node_id: NodeID,
        num_outputs: u16,
    ) -> Result<Vec<EdgeID>, ()> {
        if node_id == self.graph_out_id {
            return Err(());
        }

        let node_entry = self.nodes.get_mut(node_id.0).ok_or(())?;

        let old_num_outputs = node_entry.num_outputs;
        let mut removed_edges = Vec::new();
        if num_outputs < old_num_outputs {
            for port_idx in num_outputs..old_num_outputs {
                removed_edges
                    .append(&mut self.remove_edges_with_output_port(node_id, OutPortIdx(port_idx)));
            }
        }

        self.nodes[node_id.0].num_outputs = num_outputs;

        self.needs_compile = true;
        Ok(removed_edges)
    }

    /// Add an [Edge] (port connection) to the graph.
    ///
    /// * `src_node_id` - The ID of the source node.
    /// * `src_port_idx` - The index of the source port. This must be an output
    /// port on the source node.
    /// * `dst_node_id` - The ID of the destination node.
    /// * `dst_port_idx` - The index of the destination port. This must be an
    /// input port on the destination node.
    /// * `check_for_cycles` - If `true`, then this will run a check to
    /// see if adding this edge will create a cycle in the graph, and
    /// return an error if it does.
    ///
    /// If successful, this returns the globally unique identifier assigned
    /// to this edge.
    ///
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn add_edge(
        &mut self,
        src_node: NodeID,
        src_port: impl Into<OutPortIdx>,
        dst_node: NodeID,
        dst_port: impl Into<InPortIdx>,
        check_for_cycles: bool,
    ) -> Result<EdgeID, AddEdgeError> {
        let src_port: OutPortIdx = src_port.into();
        let dst_port: InPortIdx = dst_port.into();

        let src_node_entry = self
            .nodes
            .get(src_node.0)
            .ok_or(AddEdgeError::SrcNodeNotFound(src_node))?;
        let dst_node_entry = self
            .nodes
            .get(dst_node.0)
            .ok_or(AddEdgeError::DstNodeNotFound(dst_node))?;

        if src_port.0 >= src_node_entry.num_outputs {
            return Err(AddEdgeError::OutPortOutOfRange {
                node: src_node,
                port_idx: src_port,
                num_out_ports: src_node_entry.num_outputs,
            });
        }
        if dst_port.0 >= dst_node_entry.num_inputs {
            return Err(AddEdgeError::InPortOutOfRange {
                node: dst_node,
                port_idx: dst_port,
                num_in_ports: dst_node_entry.num_inputs,
            });
        }

        if src_node.0 == dst_node.0 {
            return Err(AddEdgeError::CycleDetected);
        }

        if !self.existing_edges.insert(EdgeHash {
            src_node,
            src_port,
            dst_node,
            dst_port,
        }) {
            return Err(AddEdgeError::EdgeAlreadyExists);
        }

        if !self.connected_input_ports.insert((dst_node, dst_port)) {
            return Err(AddEdgeError::InputPortAlreadyConnected(dst_node, dst_port));
        }

        let new_edge_id = EdgeID(self.edges.insert(Edge {
            id: EdgeID(thunderdome::Index::DANGLING),
            src_node,
            src_port,
            dst_node,
            dst_port,
        }));
        self.edges[new_edge_id.0].id = new_edge_id;

        if check_for_cycles {
            if self.cycle_detected() {
                self.edges.remove(new_edge_id.0);

                return Err(AddEdgeError::CycleDetected);
            }
        }

        self.needs_compile = true;

        Ok(new_edge_id)
    }

    /// Remove the given [Edge] (port connection) from the graph.
    ///
    /// If the edge did not exist in the graph, then `false` will be
    /// returned.
    pub fn remove_edge(&mut self, edge_id: EdgeID) -> bool {
        if let Some(edge) = self.edges.remove(edge_id.0) {
            self.existing_edges.remove(&EdgeHash {
                src_node: edge.src_node,
                src_port: edge.src_port,
                dst_node: edge.dst_node,
                dst_port: edge.dst_port,
            });
            self.connected_input_ports
                .remove(&(edge.dst_node, edge.dst_port));

            self.needs_compile = true;

            true
        } else {
            false
        }
    }

    /// Get information about the given [Edge]
    pub fn edge(&self, edge_id: EdgeID) -> Option<&Edge> {
        self.edges.get(edge_id.0)
    }

    fn compile_and_send_schedule(&mut self) -> Result<(), CompileGraphError> {
        if !self.needs_compile {
            return Ok(());
        }

        let schedule = self.compile()?;

        // TODO

        Ok(())
    }

    /// Compile the graph into a schedule.
    fn compile(&mut self) -> Result<CompiledSchedule, CompileGraphError> {
        self.needs_compile = false;

        compiler::compile(
            &mut self.nodes,
            &mut self.edges,
            self.graph_in_id,
            self.graph_out_id,
        )
    }

    /// Returns `true` if `AudioGraph::compile()` should be called
    /// again because the state of the graph has changed since the last
    /// compile.
    pub fn needs_compile(&self) -> bool {
        self.needs_compile
    }

    fn remove_edges_with_input_port(
        &mut self,
        node_id: NodeID,
        port_idx: InPortIdx,
    ) -> Vec<EdgeID> {
        let mut edges_to_remove: Vec<EdgeID> = Vec::new();

        // Remove all existing edges which have this port.
        for (edge_id, edge) in self.edges.iter() {
            if edge.dst_node == node_id && edge.dst_port == port_idx {
                edges_to_remove.push(EdgeID(edge_id));
            }
        }

        for edge_id in edges_to_remove.iter() {
            self.remove_edge(*edge_id);
        }

        edges_to_remove
    }

    fn remove_edges_with_output_port(
        &mut self,
        node_id: NodeID,
        port_idx: OutPortIdx,
    ) -> Vec<EdgeID> {
        let mut edges_to_remove: Vec<EdgeID> = Vec::new();

        // Remove all existing edges which have this port.
        for (edge_id, edge) in self.edges.iter() {
            if edge.src_node == node_id && edge.src_port == port_idx {
                edges_to_remove.push(EdgeID(edge_id));
            }
        }

        for edge_id in edges_to_remove.iter() {
            self.remove_edge(*edge_id);
        }

        edges_to_remove
    }

    fn cycle_detected(&mut self) -> bool {
        compiler::cycle_detected(
            &mut self.nodes,
            &mut self.edges,
            self.graph_in_id,
            self.graph_out_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::compiler::BufferIdx;
    use super::*;
    use ahash::AHashSet;

    // Simplest graph compile test:
    //
    //  ┌───┐  ┌───┐
    //  │ 0 ┼──► 1 │
    //  └───┘  └───┘
    #[test]
    fn simplest_graph_compile_test() {
        let mut graph = AudioGraph::with_capacity(1, 1, 32, 32);

        let node0 = graph.graph_in_node();
        let node1 = graph.graph_out_node();

        let edge0 = graph.add_edge(node0, 0, node1, 0, false).unwrap();

        let schedule = graph.compile().unwrap();

        dbg!(&schedule);

        assert_eq!(schedule.schedule.len(), 2);
        assert!(schedule.num_buffers > 0);

        // First node must be node 0
        assert_eq!(schedule.schedule[0].id, node0);
        // Last node must be node 1
        assert_eq!(schedule.schedule[1].id, node1);

        verify_node(node0, &[], &schedule, &graph);
        verify_node(node1, &[false], &schedule, &graph);

        verify_edge(edge0, &graph, &schedule);
    }

    // Graph compile test 1:
    //
    //              ┌───┐  ┌───┐
    //         ┌────►   ┼──►   │
    //       ┌─┼─┐  ┼ 3 ┼──►   │
    //   ┌───►   │  └───┘  │   │  ┌───┐
    // ┌─┼─┐ │ 1 │  ┌───┐  │ 5 ┼──►   │
    // │   │ └─┬─┘  ┼   ┼──►   ┼──► 6 │
    // │ 0 │   └────► 4 ┼──►   │  └───┘
    // └─┬─┘        └───┘  │   │
    //   │   ┌───┐         │   │
    //   └───► 2 ┼─────────►   │
    //       └───┘         └───┘
    #[test]
    fn graph_compile_test_1() {
        let mut graph = AudioGraph::with_capacity(2, 2, 32, 32);

        let node0 = graph.graph_in_node();
        let node1 = graph.add_node(1, 2, DummyAudioNode);
        let node2 = graph.add_node(1, 1, DummyAudioNode);
        let node3 = graph.add_node(2, 2, DummyAudioNode);
        let node4 = graph.add_node(2, 2, DummyAudioNode);
        let node5 = graph.add_node(5, 2, DummyAudioNode);
        let node6 = graph.graph_out_node();

        let edge0 = graph.add_edge(node0, 0, node1, 0, false).unwrap();
        let edge1 = graph.add_edge(node0, 1, node2, 0, false).unwrap();
        let edge2 = graph.add_edge(node1, 0, node3, 0, false).unwrap();
        let edge3 = graph.add_edge(node1, 1, node4, 1, false).unwrap();
        let edge4 = graph.add_edge(node3, 0, node5, 0, false).unwrap();
        let edge5 = graph.add_edge(node3, 1, node5, 1, false).unwrap();
        let edge6 = graph.add_edge(node4, 0, node5, 2, false).unwrap();
        let edge7 = graph.add_edge(node4, 1, node5, 3, false).unwrap();
        let edge8 = graph.add_edge(node2, 0, node5, 4, false).unwrap();
        let edge9 = graph.add_edge(node5, 0, node6, 0, false).unwrap();
        let edge10 = graph.add_edge(node5, 1, node6, 1, false).unwrap();

        let schedule = graph.compile().unwrap();

        dbg!(&schedule);

        assert_eq!(schedule.schedule.len(), 7);
        // Node 5 needs at-least 7 buffers
        assert!(schedule.num_buffers > 6);

        // First node must be node 0
        assert_eq!(schedule.schedule[0].id, node0);
        // Next two nodes must be 1 and 2
        assert!(schedule.schedule[1].id == node1 || schedule.schedule[1].id == node2);
        assert!(schedule.schedule[2].id == node1 || schedule.schedule[2].id == node2);
        // Next two nodes must be 3 and 4
        assert!(schedule.schedule[3].id == node3 || schedule.schedule[3].id == node4);
        assert!(schedule.schedule[4].id == node3 || schedule.schedule[4].id == node4);
        // Next node must be 5
        assert_eq!(schedule.schedule[5].id, node5);
        // Last node must be 6
        assert_eq!(schedule.schedule[6].id, node6);

        verify_node(node0, &[], &schedule, &graph);
        verify_node(node1, &[false], &schedule, &graph);
        verify_node(node2, &[false], &schedule, &graph);
        verify_node(node3, &[false, true], &schedule, &graph);
        verify_node(node4, &[true, false], &schedule, &graph);
        verify_node(
            node5,
            &[false, false, false, false, false],
            &schedule,
            &graph,
        );
        verify_node(node6, &[false, false], &schedule, &graph);

        verify_edge(edge0, &graph, &schedule);
        verify_edge(edge1, &graph, &schedule);
        verify_edge(edge2, &graph, &schedule);
        verify_edge(edge3, &graph, &schedule);
        verify_edge(edge4, &graph, &schedule);
        verify_edge(edge5, &graph, &schedule);
        verify_edge(edge6, &graph, &schedule);
        verify_edge(edge7, &graph, &schedule);
        verify_edge(edge8, &graph, &schedule);
        verify_edge(edge9, &graph, &schedule);
        verify_edge(edge10, &graph, &schedule);
    }

    // Graph compile test 2:
    //
    //          ┌───┐  ┌───┐
    //     ┌────►   ┼──►   │
    //   ┌─┼─┐  ┼ 2 ┼  ┼   │  ┌───┐
    //   |   │  └───┘  │   ┼──►   │
    //   │ 0 │  ┌───┐  │ 4 ┼  ┼ 5 │
    //   └─┬─┘  ┼   ┼  ┼   │  └───┘
    //     └────► 3 ┼──►   │  ┌───┐
    //          └───┘  │   ┼──► 6 ┼
    //   ┌───┐         │   │  └───┘
    //   ┼ 1 ┼─────────►   ┼
    //   └───┘         └───┘
    #[test]
    fn graph_compile_test_2() {
        let mut graph = AudioGraph::with_capacity(2, 2, 32, 32);

        let node0 = graph.graph_in_node();
        let node1 = graph.add_node(1, 1, DummyAudioNode);
        let node2 = graph.add_node(2, 2, DummyAudioNode);
        let node3 = graph.add_node(2, 2, DummyAudioNode);
        let node4 = graph.add_node(5, 4, DummyAudioNode);
        let node5 = graph.graph_out_node();
        let node6 = graph.add_node(1, 1, DummyAudioNode);

        let edge0 = graph.add_edge(node0, 0, node2, 0, false).unwrap();
        let edge1 = graph.add_edge(node0, 0, node3, 1, false).unwrap();
        let edge2 = graph.add_edge(node2, 0, node4, 0, false).unwrap();
        let edge3 = graph.add_edge(node3, 1, node4, 3, false).unwrap();
        let edge4 = graph.add_edge(node1, 0, node4, 4, false).unwrap();
        let edge5 = graph.add_edge(node4, 0, node5, 0, false).unwrap();
        let edge6 = graph.add_edge(node4, 2, node6, 0, false).unwrap();

        let schedule = graph.compile().unwrap();

        dbg!(&schedule);

        assert_eq!(schedule.schedule.len(), 7);
        // Node 4 needs at-least 8 buffers
        assert!(schedule.num_buffers > 7);

        // First two nodes must be 1 and 2
        assert!(schedule.schedule[0].id == node0 || schedule.schedule[0].id == node1);
        assert!(schedule.schedule[1].id == node0 || schedule.schedule[1].id == node1);
        // Next two nodes must be 2 and 3
        assert!(schedule.schedule[2].id == node2 || schedule.schedule[2].id == node3);
        assert!(schedule.schedule[3].id == node2 || schedule.schedule[3].id == node3);
        // Next node must be 4
        assert_eq!(schedule.schedule[4].id, node4);
        // Last two nodes must be 5 and 6
        assert!(schedule.schedule[5].id == node5 || schedule.schedule[5].id == node6);
        assert!(schedule.schedule[6].id == node5 || schedule.schedule[6].id == node6);

        verify_edge(edge0, &graph, &schedule);
        verify_edge(edge1, &graph, &schedule);
        verify_edge(edge2, &graph, &schedule);
        verify_edge(edge3, &graph, &schedule);
        verify_edge(edge4, &graph, &schedule);
        verify_edge(edge5, &graph, &schedule);
        verify_edge(edge6, &graph, &schedule);

        verify_node(node0, &[], &schedule, &graph);
        verify_node(node1, &[true], &schedule, &graph);
        verify_node(node2, &[false, true], &schedule, &graph);
        verify_node(node3, &[true, false], &schedule, &graph);
        verify_node(node4, &[false, true, true, false, false], &schedule, &graph);
        verify_node(node5, &[false, true], &schedule, &graph);
        verify_node(node6, &[false], &schedule, &graph);
    }

    fn verify_node(
        node_id: NodeID,
        in_ports_that_should_clear: &[bool],
        schedule: &CompiledSchedule,
        graph: &AudioGraph,
    ) {
        let node = graph.node_info(node_id).unwrap();
        let scheduled_node = schedule.schedule.iter().find(|&s| s.id == node_id).unwrap();

        assert_eq!(scheduled_node.id, node_id);
        assert_eq!(scheduled_node.input_buffers.len(), node.num_inputs as usize);
        assert_eq!(
            scheduled_node.output_buffers.len(),
            node.num_outputs as usize
        );

        assert_eq!(in_ports_that_should_clear.len(), node.num_inputs as usize);

        for (buffer, should_clear) in scheduled_node
            .input_buffers
            .iter()
            .zip(in_ports_that_should_clear)
        {
            assert_eq!(buffer.should_clear, *should_clear);
        }

        let mut buffer_alias_check: AHashSet<BufferIdx> = AHashSet::default();

        for buffer in scheduled_node.input_buffers.iter() {
            assert!(buffer_alias_check.insert(buffer.buffer_index));
        }

        for buffer in scheduled_node.output_buffers.iter() {
            assert!(buffer_alias_check.insert(buffer.buffer_index));
        }
    }

    fn verify_edge(edge_id: EdgeID, graph: &AudioGraph, schedule: &CompiledSchedule) {
        let edge = graph.edge(edge_id).unwrap();

        let mut src_buffer_idx = None;
        let mut dst_buffer_idx = None;
        for node in schedule.schedule.iter() {
            if node.id == edge.src_node {
                src_buffer_idx = Some(node.output_buffers[edge.src_port.0 as usize].buffer_index);
                if dst_buffer_idx.is_some() {
                    break;
                }
            } else if node.id == edge.dst_node {
                dst_buffer_idx = Some(node.input_buffers[edge.dst_port.0 as usize].buffer_index);
                if src_buffer_idx.is_some() {
                    break;
                }
            }
        }

        let src_buffer_idx = src_buffer_idx.unwrap();
        let dst_buffer_idx = dst_buffer_idx.unwrap();

        assert_eq!(src_buffer_idx, dst_buffer_idx);
    }

    #[test]
    fn many_to_one_detection() {
        let mut graph = AudioGraph::with_capacity(2, 1, 32, 32);

        let node1 = graph.graph_in_node();
        let node2 = graph.graph_out_node();

        graph.add_edge(node1, 0, node2, 0, false).unwrap();

        if let Err(AddEdgeError::InputPortAlreadyConnected(node_id, port_id)) =
            graph.add_edge(node1, OutPortIdx(1), node2, InPortIdx(0), false)
        {
            assert_eq!(node_id, node2);
            assert_eq!(port_id, InPortIdx(0));
        } else {
            panic!("expected error");
        }
    }

    #[test]
    fn cycle_detection() {
        let mut graph = AudioGraph::with_capacity(0, 2, 32, 32);

        let node1 = graph.add_node(1, 1, DummyAudioNode);
        let node2 = graph.add_node(2, 1, DummyAudioNode);
        let node3 = graph.add_node(1, 1, DummyAudioNode);

        graph.add_edge(node1, 0, node2, 0, false).unwrap();
        graph.add_edge(node2, 0, node3, 0, false).unwrap();
        let edge3 = graph.add_edge(node3, 0, node1, 0, false).unwrap();

        assert!(graph.cycle_detected());

        graph.remove_edge(edge3);

        assert!(!graph.cycle_detected());

        graph.add_edge(node3, 0, node2, 1, false).unwrap();

        assert!(graph.cycle_detected());
    }
}
