extern crate conllx;

#[macro_use]
extern crate failure;

extern crate hdf5;

extern crate itertools;

extern crate protobuf;

extern crate serde;

#[macro_use]
extern crate serde_derive;

extern crate tensorflow as tf;

extern crate tf_embed;

extern crate tf_proto;

mod input;
pub use input::{LayerEmbeddings, SentVec, SentVectorizer};

mod numberer;
pub use numberer::Numberer;

mod tag;
pub use tag::Tag;

pub mod tensorflow;

mod writer;
pub use writer::{Collector, HDF5Collector};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
