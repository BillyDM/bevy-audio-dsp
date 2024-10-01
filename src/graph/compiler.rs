use smallvec::SmallVec;
use std::{collections::VecDeque, rc::Rc};
use thunderdome::Arena;

use super::error::CompileGraphError;

pub struct NodeEntry<N> {
    pub id: NodeID,
    /// The number of input ports used by the node
    pub num_inputs: u16,
    /// The number of output ports used by the node
    pub num_outputs: u16,
    pub weight: N,
    /// The edges connected to this node's input ports.
    incoming: SmallVec<[Edge; 4]>,
    /// The edges connected to this node's output ports.
    outgoing: SmallVec<[Edge; 4]>,
}

impl<N> NodeEntry<N> {
    pub fn new(num_inputs: u16, num_outputs: u16, weight: N) -> Self {
        Self {
            id: NodeID(thunderdome::Index::DANGLING),
            num_inputs,
            num_outputs,
            weight,
            incoming: SmallVec::new(),
            outgoing: SmallVec::new(),
        }
    }
}

/// A globally unique identifier for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeID(pub(super) thunderdome::Index);

/// The index for an input port on a particular [Node].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InPortIdx(pub u16);

impl From<u16> for InPortIdx {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl Into<usize> for InPortIdx {
    fn into(self) -> usize {
        usize::from(self.0)
    }
}

/// The index for an output port on a particular [Node].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutPortIdx(pub u16);

impl From<u16> for OutPortIdx {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl Into<usize> for OutPortIdx {
    fn into(self) -> usize {
        usize::from(self.0)
    }
}

/// The index of the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferIdx(pub u32);

impl Into<usize> for BufferIdx {
    fn into(self) -> usize {
        self.0 as usize
    }
}

/// A globally unique identifier for an [Edge].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeID(pub(super) thunderdome::Index);

/// An [Edge] is a connection from source node and port to a
/// destination node and port.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Edge {
    pub id: EdgeID,
    /// The ID of the source node used by this edge.
    pub src_node: NodeID,
    /// The ID of the source port used by this edge.
    pub src_port: OutPortIdx,
    /// The ID of the destination node used by this edge.
    pub dst_node: NodeID,
    /// The ID of the destination port used by this edge.
    pub dst_port: InPortIdx,
}

/// A [CompiledSchedule] is the output of the graph compiler.
#[derive(Clone, Debug)]
pub struct CompiledSchedule {
    /// A list of nodes to evaluate in order to render audio,
    /// in topological order.
    pub schedule: Vec<ScheduledNode>,
    /// The total number of buffers required to allocate
    pub num_buffers: usize,
}

/// A [ScheduledNode] is a [Node] that has been assigned buffers
/// and a place in the schedule.
#[derive(Clone, Debug)]
pub struct ScheduledNode {
    /// The node ID
    pub id: NodeID,

    /// The assigned input buffers.
    pub input_buffers: SmallVec<[InBufferAssignment; 4]>,
    /// The assigned output buffers.
    pub output_buffers: SmallVec<[OutBufferAssignment; 4]>,
}

impl ScheduledNode {
    pub fn new(id: NodeID) -> Self {
        Self {
            id,
            input_buffers: SmallVec::new(),
            output_buffers: SmallVec::new(),
        }
    }
}

/// Represents a single buffer assigned to an input port
#[derive(Copy, Clone, Debug)]
pub struct InBufferAssignment {
    /// The index of the buffer assigned
    pub buffer_index: BufferIdx,
    /// Whether the engine should clear the buffer before
    /// passing it to a process
    pub should_clear: bool,
    /// Buffers are reused, the "generation" represents
    /// how many times this buffer has been used before
    /// this assignment. Kept for debugging and visualization.
    pub generation: usize,
}

/// Represents a single buffer assigned to an output port
#[derive(Copy, Clone, Debug)]
pub struct OutBufferAssignment {
    /// The index of the buffer assigned
    pub buffer_index: BufferIdx,
    /// Buffers are reused, the "generation" represents
    /// how many times this buffer has been used before
    /// this assignment. Kept for debugging and visualization.
    pub generation: usize,
}

/// A reference to an abstract buffer during buffer allocation.
#[derive(Debug, Clone, Copy)]
struct BufferRef {
    /// The index of the buffer
    idx: BufferIdx,
    /// The generation, or the nth time this buffer has
    /// been assigned to a different edge in the graph.
    generation: usize,
}

/// An allocator for managing and reusing [BufferRef]s.
#[derive(Debug, Clone)]
struct BufferAllocator {
    /// A list of free buffers that may be reallocated
    free_list: Vec<BufferRef>,
    /// The maximum number of buffers used
    count: u32,
}

impl BufferAllocator {
    /// Create a new allocator, `num_types` defines the number
    /// of buffer types we may allocate.
    fn new(initial_capacity: usize) -> Self {
        Self {
            free_list: Vec::with_capacity(initial_capacity),
            count: 0,
        }
    }

    /// Acquire a new buffer
    fn acquire(&mut self) -> Rc<BufferRef> {
        let entry = self.free_list.pop().unwrap_or_else(|| {
            let idx = self.count;
            self.count += 1;
            BufferRef {
                idx: BufferIdx(idx),
                generation: 0,
            }
        });
        Rc::new(BufferRef {
            idx: entry.idx,
            generation: entry.generation,
        })
    }

    /// Release a BufferRef
    fn release(&mut self, buffer_ref: Rc<BufferRef>) {
        if Rc::strong_count(&buffer_ref) == 1 {
            self.free_list.push(BufferRef {
                idx: buffer_ref.idx,
                generation: buffer_ref.generation + 1,
            });
        }
    }

    /// Consume the allocator to return the maximum number of buffers used
    fn num_buffers(self) -> u32 {
        self.count
    }
}

/// Main compilation algorithm
pub fn compile<'a, N>(
    nodes: &mut Arena<NodeEntry<N>>,
    edges: &mut Arena<Edge>,
) -> Result<CompiledSchedule, CompileGraphError> {
    Ok(GraphIR::preprocess(nodes, edges)
        .sort_topologically(true)?
        .solve_buffer_requirements()?
        .merge())
}

pub fn cycle_detected<'a, N>(
    nodes: &'a mut Arena<NodeEntry<N>>,
    edges: &'a mut Arena<Edge>,
) -> bool {
    if let Err(CompileGraphError::CycleDetected) =
        GraphIR::<N>::preprocess(nodes, edges).sort_topologically(false)
    {
        true
    } else {
        false
    }
}

/// Internal IR used by the compiler algorithm. Built incrementally
/// via the compiler passes.
struct GraphIR<'a, N> {
    nodes: &'a mut Arena<NodeEntry<N>>,
    edges: &'a mut Arena<Edge>,

    /// The topologically sorted schedule of the graph. Built internally.
    schedule: Vec<ScheduledNode>,
    /// The maximum number of buffers used.
    max_num_buffers: usize,
}

impl<'a, N> GraphIR<'a, N> {
    /// Construct a [GraphIR] instance from lists of nodes and edges, building
    /// up the adjacency table and creating an empty schedule.
    fn preprocess(nodes: &'a mut Arena<NodeEntry<N>>, edges: &'a mut Arena<Edge>) -> Self {
        for (_, node) in nodes.iter_mut() {
            node.incoming.clear();
            node.outgoing.clear();
        }

        for (_, edge) in edges.iter() {
            nodes[edge.src_node.0].outgoing.push(*edge);
            nodes[edge.dst_node.0].incoming.push(*edge);
        }

        Self {
            nodes,
            edges,
            schedule: vec![],
            max_num_buffers: 0,
        }
    }

    /// Sort the nodes topologically using Kahn's algorithm.
    /// https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
    fn sort_topologically(mut self, build_schedule: bool) -> Result<Self, CompileGraphError> {
        let mut in_degree = vec![0i32; self.nodes.capacity()];
        let mut queue = VecDeque::with_capacity(self.nodes.len());

        if build_schedule {
            self.schedule.reserve(self.nodes.len());
        }

        let mut num_visited = 0;

        // Calculate in-degree of each vertex
        for (_, node_entry) in self.nodes.iter() {
            for edge in node_entry.outgoing.iter() {
                in_degree[edge.dst_node.0.slot() as usize] += 1;
            }
        }

        // Enqueue vertices with 0 in-degree
        for (node_id, _) in self.nodes.iter() {
            if in_degree[node_id.slot() as usize] == 0 {
                queue.push_back(node_id);
            }
        }

        // BFS traversal
        while let Some(node_id) = queue.pop_front() {
            num_visited += 1;

            let node_entry = &self.nodes[node_id];

            // Reduce in-degree of adjacent vertices
            for edge in node_entry.outgoing.iter() {
                in_degree[edge.dst_node.0.slot() as usize] -= 1;

                // If in-degree becomes 0, enqueue it
                if in_degree[edge.dst_node.0.slot() as usize] == 0 {
                    queue.push_back(edge.dst_node.0);
                }
            }

            if build_schedule {
                self.schedule.push(ScheduledNode::new(NodeID(node_id)));
            }
        }

        // If not all vertices are visited, cycle
        if num_visited != self.nodes.len() {
            return Err(CompileGraphError::CycleDetected);
        }

        Ok(self)
    }

    /*
    /// Walk the nodes of the graph and add them to the schedule.
    fn sort_topologically(mut self) -> Result<Self, CompileGraphError> {
        if self.tarjan() != 0 {
            return Err(CompileGraphError::CycleDetected);
        }

        let mut queue = VecDeque::with_capacity(self.nodes.len());
        self.schedule.reserve(self.nodes.len());

        // Initialize the queue with roots
        for node in self
            .nodes
            .iter()
            .map(|(_, n)| n)
            .filter(|n| n.incoming.is_empty())
        {
            queue.push_back(node);
        }

        let mut visited = Arena::<()>::with_capacity(self.nodes.capacity());

        while let Some(node) = queue.pop_front() {
            for next_node_id in node.outgoing.iter().map(|e| e.dst_node) {
                if visited.insert_at(next_node_id.0, ()).is_none() {
                    queue.push_back(&self.nodes[next_node_id.0]);
                }
            }

            self.schedule.push(ScheduledNode::new(node.id));
        }

        Ok(self)
    }
    */

    fn solve_buffer_requirements(mut self) -> Result<Self, CompileGraphError> {
        let mut allocator = BufferAllocator::new(64);
        let mut assignment_table: Arena<Rc<BufferRef>> =
            Arena::with_capacity(self.edges.capacity());
        let mut buffers_to_release: Vec<Rc<BufferRef>> = Vec::with_capacity(64);

        for entry in &mut self.schedule {
            // Collect the inputs to the algorithm, the incoming/outgoing edges of this node.

            let node_entry = &self.nodes[entry.id.0];

            buffers_to_release.clear();
            if buffers_to_release.capacity()
                < node_entry.num_inputs as usize + node_entry.num_outputs as usize
            {
                buffers_to_release.reserve(
                    node_entry.num_inputs as usize + node_entry.num_outputs as usize
                        - buffers_to_release.capacity(),
                );
            }

            entry
                .input_buffers
                .reserve_exact(node_entry.num_inputs as usize);
            entry
                .output_buffers
                .reserve_exact(node_entry.num_outputs as usize);

            for port_idx in 0..node_entry.num_inputs {
                let port_idx = InPortIdx(port_idx);

                let edges: SmallVec<[&Edge; 4]> = node_entry
                    .incoming
                    .iter()
                    .filter(|edge| edge.dst_port == port_idx)
                    .collect();

                if edges.is_empty() {
                    // Case 1: The port is an input and it is unconnected. Acquire a buffer, and
                    //         assign it. The buffer must be cleared. Release the buffer once the
                    //         node assignments are done.
                    let buffer = allocator.acquire();
                    entry.input_buffers.push(InBufferAssignment {
                        buffer_index: buffer.idx,
                        generation: buffer.generation,
                        should_clear: true,
                    });
                    buffers_to_release.push(buffer);
                } else if edges.len() == 1 {
                    // Case 2: The port is an input, and has exactly one incoming edge. Lookup the
                    //         corresponding buffer and assign it. Buffer should not be cleared.
                    //         Release the buffer once the node assignments are done.
                    let buffer = assignment_table
                        .remove(edges[0].id.0)
                        .expect("No buffer assigned to edge!");
                    entry.input_buffers.push(InBufferAssignment {
                        buffer_index: buffer.idx,
                        generation: buffer.generation,
                        should_clear: false,
                    });
                    buffers_to_release.push(buffer);
                } else {
                    return Err(CompileGraphError::ManyToOneError(entry.id, port_idx));
                }
            }

            for port_idx in 0..node_entry.num_outputs {
                let port_idx = OutPortIdx(port_idx);

                let edges: SmallVec<[&Edge; 4]> = node_entry
                    .outgoing
                    .iter()
                    .filter(|edge| edge.src_port == port_idx)
                    .collect();

                if edges.is_empty() {
                    // Case 1: The port is an output and it is unconnected. Acquire a buffer and
                    //         assign it. The buffer does not need to be cleared. Release the
                    //         buffer once the node assignments are done.
                    let buffer = allocator.acquire();
                    entry.output_buffers.push(OutBufferAssignment {
                        buffer_index: buffer.idx,
                        generation: buffer.generation,
                    });
                    buffers_to_release.push(buffer);
                } else {
                    // Case 2: The port is an output. Acquire a buffer, and add to the assignment
                    //         table with any corresponding edge IDs. For each edge, update the
                    //         assigned buffer table. Buffer should not be cleared or released.
                    let buffer = allocator.acquire();
                    for edge in &edges {
                        assignment_table.insert_at(edge.id.0, Rc::clone(&buffer));
                    }
                    entry.output_buffers.push(OutBufferAssignment {
                        buffer_index: buffer.idx,
                        generation: buffer.generation,
                    });
                }
            }

            for buffer in buffers_to_release.drain(..) {
                allocator.release(buffer);
            }
        }

        self.max_num_buffers = allocator.num_buffers() as usize;
        Ok(self)
    }

    /// Merge the GraphIR into a [CompiledSchedule].
    fn merge(self) -> CompiledSchedule {
        CompiledSchedule {
            schedule: self.schedule,
            num_buffers: self.max_num_buffers,
        }
    }
}
