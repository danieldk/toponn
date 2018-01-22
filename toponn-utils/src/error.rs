use std::io;

use conllx;
use tf_embed;
use toponn;
use serde_cbor;
use toml;

error_chain! {
    foreign_links {
        Conllx(conllx::Error);
        IO(io::Error);
        TopoNN(toponn::Error);
        TOML(toml::de::Error);
        TOMLSerde(toml::ser::Error);
        CBORSerde(serde_cbor::Error);
        Embed(tf_embed::Error);
    }

    errors {
        ConfigError(m: String) {
            description("configuration error")
                display("configuration error: {}", m)
        }
    }
}
