pub trait AudioBackend {
    type StreamHandle;
    type Stream: Send;
}
