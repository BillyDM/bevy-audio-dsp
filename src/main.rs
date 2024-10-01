mod buffer;
pub mod graph;
mod silence_mask;
pub mod util;

pub use silence_mask::SilenceMask;

/// The maximum number of frames that can appear in a processing
/// block.
///
/// This number is a balance between processing overhead and
/// cache efficiency. Lower values have better cache efficieny
/// but more overhead, and higher values have worse cache
/// efficiency but less overhead. We may need to experiment with
/// different values to see what is the best for a typical game
/// audio graph. (The value must also be a power of two.)
pub const MAX_BLOCK_FRAMES: usize = 256;

pub fn main() {
    println!("Hello World!");
}
