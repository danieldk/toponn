use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use failure::Error;
use ordered_float::NotNan;
use tf_embed;
use tf_embed::ReadWord2Vec;

use toponn::tensorflow::{Model, PlateauLearningRate};
use toponn::LayerEmbeddings;

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub labeler: Labeler,
    pub embeddings: Embeddings,
    pub model: Model,
    pub train: Train,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        self.embeddings.word.filename =
            relativize_path(config_path, &self.embeddings.word.filename)?;
        self.embeddings.tag.filename = relativize_path(config_path, &self.embeddings.tag.filename)?;
        self.model.graph = relativize_path(config_path, &self.model.graph)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;

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
    pub fn load_embeddings(&self) -> Result<LayerEmbeddings, Error> {
        let token_embeddings = self.load_layer_embeddings(&self.word)?;
        let tag_embeddings = self.load_layer_embeddings(&self.tag)?;

        Ok(LayerEmbeddings::new(token_embeddings, tag_embeddings))
    }

    pub fn load_layer_embeddings(
        &self,
        embeddings: &Embedding,
    ) -> Result<tf_embed::Embeddings, Error> {
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

fn relativize_path(config_path: &Path, filename: &str) -> Result<String, Error> {
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
        .ok_or(format_err!(
            "Cannot get parent path of the configuration file: {}",
            abs_config_path.to_string_lossy()
        ))?.join(path)
        .to_str()
        .ok_or(format_err!(
            "Cannot cannot convert partent path to string: {}",
            abs_config_path.to_string_lossy()
        ))?.to_owned())
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Train {
    pub initial_lr: NotNan<f32>,
    pub lr_scale: NotNan<f32>,
    pub lr_patience: usize,
    pub patience: usize,
}

impl Train {
    pub fn lr_schedule(&self) -> PlateauLearningRate {
        PlateauLearningRate::new(
            self.initial_lr.into_inner(),
            self.lr_scale.into_inner(),
            self.lr_patience,
        )
    }
}
