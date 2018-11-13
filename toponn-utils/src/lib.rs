extern crate conllx;

#[macro_use]
extern crate failure;

extern crate tf_embed;

extern crate toponn;

extern crate serde;

extern crate serde_cbor;

#[macro_use]
extern crate serde_derive;

extern crate toml;

mod config;
pub use config::{Config, Embedding, Embeddings, Labeler};

mod serialization;
pub use serialization::{CborRead, CborWrite, TomlRead};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod config_tests;
