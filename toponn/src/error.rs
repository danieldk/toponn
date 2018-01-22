use std::error::Error as StdError;
use std::io;

use hdf5;
use protobuf::ProtobufError;
use tf;

error_chain! {
    foreign_links {
        HDF5(hdf5::Error);
        IO(io::Error);
        Protobuf(ProtobufError);
    }

    errors {
    IncorrectBatchSize(batch_size: usize, correct_batch_size: usize) {
        description("incorrect batch size")
            display("incorrect batch size, was: {}, should be: <= {}", batch_size, correct_batch_size)
    }
    MissingTopologicalField(t: String) {
        description("topological field feature missing")
            display("topological field feature missing: '{}'", t)
    }
    MissingPOSTag(t: String) {
        description("part-of-speech tag missing")
            display("part-of-speech tag missing: '{}'", t)
    }
        TensorflowError(t: String) {
            description("tensorflow error")
            display("tensorflow error: '{}'", t)
        }
    }
}

impl From<tf::Status> for Error {
    fn from(status: tf::Status) -> Self {
        ErrorKind::TensorflowError(status.description().to_owned()).into()
    }
}
