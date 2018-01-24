use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tf_embed;
use tf_embed::ReadWord2Vec;

use toponn::LayerEmbeddings;
use toponn::tensorflow::ModelConfig;

use {ErrorKind, Result};

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub labeler: Labeler,
    pub embeddings: Embeddings,
    #[serde(rename = "model")] pub model: ModelConfig,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        self.embeddings.word.filename =
            relativize_path(config_path, &self.embeddings.word.filename)?;
        self.embeddings.tag.filename = relativize_path(config_path, &self.embeddings.tag.filename)?;
        self.model.filename = relativize_path(config_path, &self.model.filename)?;

        Ok(())
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Embeddings {
    pub word: Embedding,
    pub tag: Embedding,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Embedding {
    pub filename: String,
    pub normalize: bool,
}

impl Embeddings {
    pub fn load_embeddings(&self) -> Result<LayerEmbeddings> {
        let token_embeddings = self.load_layer_embeddings(&self.word)?;
        let tag_embeddings = self.load_layer_embeddings(&self.tag)?;

        Ok(LayerEmbeddings::new(token_embeddings, tag_embeddings))
    }

    pub fn load_layer_embeddings(&self, embeddings: &Embedding) -> Result<tf_embed::Embeddings> {
        let f = File::open(&embeddings.filename)?;
        let mut embeds = tf_embed::Embeddings::read_word2vec_binary(&mut BufReader::new(f))?;

        if embeddings.normalize {
            embeds.normalize()
        }

        Ok(embeds)
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Labeler {
    pub labels: String,
}

fn relativize_path(config_path: &Path, filename: &str) -> Result<String> {
    if filename.is_empty() {
        return Ok(filename.to_owned());
    }

    let path = Path::new(&filename);

    // Don't touch absolute paths.
    if path.is_absolute() {
        return Ok(filename.to_owned());
    }

    let abs_config_path = config_path.canonicalize()?;
    Ok(abs_config_path
        .parent()
        .ok_or(ErrorKind::ConfigError(String::from(
            "Cannot get the parent path of the configuration file",
        )))?
        .join(path)
        .to_str()
        .ok_or(ErrorKind::ConfigError(String::from(
            "Cannot convert path to string",
        )))?
        .to_owned())
}
