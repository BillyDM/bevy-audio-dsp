mod compiler;
mod error;
use ahash::AHashSet;
use thunderdome::Arena;

use compiler::{
    compile, BufferIdx, CompiledSchedule, InBufferAssignment, OutBufferAssignment, ScheduledNode,
};

pub use compiler::{Edge, EdgeID, InPortIdx, NodeEntry, NodeID, OutPortIdx};
pub use error::AddEdgeError;

pub struct NodeWeight {}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct EdgeHash {
    pub src_node: NodeID,
    pub src_port: OutPortIdx,
    pub dst_node: NodeID,
    pub dst_port: InPortIdx,
}

/// A helper struct to construct and modify audio graphs.
pub struct AudioGraph {
    nodes: Arena<NodeEntry<NodeWeight>>,
    edges: Arena<Edge>,
    connected_input_ports: AHashSet<(NodeID, InPortIdx)>,
    existing_edges: AHashSet<EdgeHash>,

    needs_compile: bool,
}

impl AudioGraph {
    /// Construct a new [AudioGraph].
    pub fn new() -> Self {
        Self {
            nodes: Arena::new(),
            edges: Arena::new(),
            connected_input_ports: AHashSet::new(),
            existing_edges: AHashSet::new(),
            needs_compile: false,
        }
    }

    /// Construct a new [AudioGraph] with some initial allocated capacity.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            nodes: Arena::with_capacity(node_capacity),
            edges: Arena::with_capacity(edge_capacity),
            connected_input_ports: AHashSet::with_capacity(edge_capacity),
            existing_edges: AHashSet::with_capacity(edge_capacity),
            needs_compile: false,
        }
    }

    /// Add a new [Node] the the audio graph.
    ///
    /// This will return the globally unique ID assigned to this node.
    pub fn add_node(&mut self, num_inputs: u16, num_outputs: u16) -> NodeID {
        self.needs_compile = true;

        let new_id = NodeID(self.nodes.insert(NodeEntry::new(
            num_inputs,
            num_outputs,
            NodeWeight {},
        )));
        self.nodes[new_id.0].id = new_id;

        new_id
    }

    /// Get info about a node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node(&self, node_id: NodeID) -> Option<&NodeEntry<NodeWeight>> {
        self.nodes.get(node_id.0)
    }

    /// Get an immutable reference to the node weight.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node_weight(&self, node_id: NodeID) -> Option<&NodeWeight> {
        self.nodes.get(node_id.0).map(|n| &n.weight)
    }

    /// Get a mutable reference to the node weight.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node_weight_mut(&mut self, node_id: NodeID) -> Option<&mut NodeWeight> {
        self.nodes.get_mut(node_id.0).map(|n| &mut n.weight)
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
    /// exist in the graph.
    pub fn remove_node(&mut self, node_id: NodeID) -> Result<(NodeWeight, Vec<EdgeID>), ()> {
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
    pub fn nodes<'a>(&'a self) -> impl Iterator<Item = &'a NodeEntry<NodeWeight>> {
        self.nodes.iter().map(|(_, n)| n)
    }

    /// Get a list of all the existing edges in the graph.
    pub fn edges<'a>(&'a self) -> impl Iterator<Item = &'a Edge> {
        self.edges.iter().map(|(_, e)| e)
    }

    /// Set the number of input ports for a particular node in the graph.
    ///
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn set_num_inputs(&mut self, node_id: NodeID, num_inputs: u16) -> Result<Vec<EdgeID>, ()> {
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
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn set_num_outputs(
        &mut self,
        node_id: NodeID,
        num_outputs: u16,
    ) -> Result<Vec<EdgeID>, ()> {
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
        src_port: OutPortIdx,
        dst_node: NodeID,
        dst_port: InPortIdx,
        check_for_cycles: bool,
    ) -> Result<EdgeID, AddEdgeError> {
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

    /// Compile the graph into a schedule.
    fn compile(&mut self) -> Result<CompiledSchedule, self::error::CompileGraphError> {
        self.needs_compile = false;

        compiler::compile(&mut self.nodes, &mut self.edges)
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
        compiler::cycle_detected(&mut self.nodes, &mut self.edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashSet;

    #[test]
    fn simplest_graph_compile_test() {
        let mut graph = AudioGraph::new();

        let node1 = graph.add_node(1, 1);
        let node2 = graph.add_node(1, 1);

        graph
            .add_edge(node1, OutPortIdx(0), node2, InPortIdx(0), false)
            .unwrap();

        let schedule = graph.compile().unwrap();

        dbg!(&schedule);

        assert_eq!(schedule.schedule.len(), 2);
        assert!(schedule.num_buffers > 0);

        verify_scheduled_node(&graph, &schedule.schedule[0], node1, &[true]);
        let edge_src_buffer_idx = schedule.schedule[0].output_buffers[0].buffer_index;

        verify_scheduled_node(&graph, &schedule.schedule[1], node2, &[false]);
        let edge_dst_buffer_idx = schedule.schedule[1].input_buffers[0].buffer_index;

        assert_eq!(edge_src_buffer_idx, edge_dst_buffer_idx);
    }

    // TODO: tests of more complex graphs

    fn verify_scheduled_node(
        graph: &AudioGraph,
        scheduled_node: &ScheduledNode,
        node_id: NodeID,
        in_ports_that_should_clear: &[bool],
    ) {
        let node = graph.node(node_id).unwrap();

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

    #[test]
    fn many_to_one_detection() {
        let mut graph = AudioGraph::new();

        let node1 = graph.add_node(0, 2);
        let node2 = graph.add_node(1, 0);

        graph
            .add_edge(node1, OutPortIdx(0), node2, InPortIdx(0), false)
            .unwrap();

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
        let mut graph = AudioGraph::new();

        let node1 = graph.add_node(1, 1);
        let node2 = graph.add_node(2, 1);
        let node3 = graph.add_node(1, 1);

        graph
            .add_edge(node1, OutPortIdx(0), node2, InPortIdx(0), false)
            .unwrap();
        graph
            .add_edge(node2, OutPortIdx(0), node3, InPortIdx(0), false)
            .unwrap();
        let edge3 = graph
            .add_edge(node3, OutPortIdx(0), node1, InPortIdx(0), false)
            .unwrap();

        assert!(graph.cycle_detected());

        graph.remove_edge(edge3);

        assert!(!graph.cycle_detected());

        graph
            .add_edge(node3, OutPortIdx(0), node2, InPortIdx(1), false)
            .unwrap();

        assert!(graph.cycle_detected());
    }
}
