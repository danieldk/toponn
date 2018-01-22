use std::fs::File;

use toponn::tensorflow::{Model, OpNames};

use super::{Config, Embedding, Embeddings, Labeler, TomlRead};

lazy_static!{
    static ref BASIC_LABELER_CHECK: Config = Config {
        labeler: Labeler {
            labels: "topo.labels".to_owned(),
        },
        embeddings: Embeddings{
            word: Embedding{
                filename: "word-vectors-null.bin".to_owned(),
                    normalize: true
            },
            tag: Embedding{
                filename: "tag-vectors-null.bin".to_owned(),
                    normalize: true
            },
        },
        model: Model {
            batch_size: 128,
            filename: "model-99.bin".to_owned(),
            intra_op_parallelism_threads: 4,
            inter_op_parallelism_threads: 4,
            op_names: OpNames {
               tokens_op: "prediction/model/tokens".to_owned(),
                tags_op: "prediction/model/tags".to_owned(),
  seq_lens_op: "prediction/model/seq_lens".to_owned(),
  token_embeds_op: "prediction/model/token_embeds".to_owned(),
  tag_embeds_op: "prediction/model/tag_embeds".to_owned(),
  predicted_op: "prediction/model/predicted".to_owned(),
            },
        }
    };
}

#[test]
fn test_parse_config() {
    let f = File::open("testdata/topo.conf").unwrap();
    let config = Config::from_toml_read(f).unwrap();
    assert_eq!(*BASIC_LABELER_CHECK, config);
}
