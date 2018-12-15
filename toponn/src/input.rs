use std::borrow::Cow;

use conllx::Sentence;
use failure::{format_err, Error};

pub enum Embeddings {
    FinalFrontier(finalfrontier::Model),
    Word2Vec(rust2vec::Embeddings),
}

impl Embeddings {
    pub fn dims(&self) -> usize {
        match self {
            Embeddings::FinalFrontier(model) => model.config().dims as usize,
            Embeddings::Word2Vec(embeds) => embeds.embed_len(),
        }
    }

    pub fn embedding(&self, word: &str) -> Cow<[f32]> {
        let embed = match self {
            Embeddings::FinalFrontier(model) => Cow::Owned(
                model
                    .embedding(word)
                    .expect("Cannot retrieve embedding.")
                    .into_raw_vec(),
            ),
            Embeddings::Word2Vec(embeds) => Cow::Borrowed(
                embeds
                    .embedding(word)
                    .or(embeds.embedding("<UNKNOWN-TOKEN>"))
                    .expect("No unknown token embedding: <UNKNOWN_TOKEN>")
                    .into_slice()
                    .expect("Non-contiguous word embedding"),
            ),
        };

        embed
    }
}

/// Sentence represented as a vector.
///
/// This data type represents a sentence as vectors (`Vec`) of tokens and
/// part-of-speech embeddings. Such a vector is typically the input to a
/// sequence labeling graph.
pub struct SentVec {
    pub tokens: Vec<f32>,
    pub tags: Vec<f32>,
}

impl SentVec {
    /// Construct a new sentence vector.
    pub fn new() -> Self {
        SentVec {
            tokens: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Construct a sentence vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        SentVec {
            tokens: Vec::with_capacity(capacity),
            tags: Vec::with_capacity(capacity),
        }
    }

    /// Decompose the sentence vector into vectors of token and
    /// part-of-speech tag embeddings.
    pub fn to_parts(self) -> (Vec<f32>, Vec<f32>) {
        (self.tokens, self.tags)
    }
}

/// Embeddings for annotation layers.
///
/// This data structure bundles embedding matrices for the input
/// annotation layers: tokens and part-of-speech.
pub struct LayerEmbeddings {
    token_embeddings: Embeddings,
    tag_embeddings: Embeddings,
}

impl LayerEmbeddings {
    /// Construct `LayerEmbeddings` from the given embeddings.
    pub fn new(token_embeddings: Embeddings, tag_embeddings: Embeddings) -> Self {
        LayerEmbeddings {
            token_embeddings: token_embeddings,
            tag_embeddings: tag_embeddings,
        }
    }

    /// Get the tag embedding matrix.
    pub fn tag_embeddings(&self) -> &Embeddings {
        &self.tag_embeddings
    }

    /// Get the token embedding matrix.
    pub fn token_embeddings(&self) -> &Embeddings {
        &self.token_embeddings
    }
}

/// Vectorizer for sentences.
///
/// An `SentVectorizer` vectorizes sentences, by replacing words/tags by
/// their indices in embedding matrices.
pub struct SentVectorizer {
    layer_embeddings: LayerEmbeddings,
}

impl SentVectorizer {
    /// Construct an input vectorizer.
    ///
    /// The vectorizer is constructed from the embedding matrices. The layer
    /// embeddings are used to find the indices into the embedding matrix for
    /// layer values.
    pub fn new(layer_embeddings: LayerEmbeddings) -> Self {
        SentVectorizer {
            layer_embeddings: layer_embeddings,
        }
    }

    /// Get the layer embeddings.
    pub fn layer_embeddings(&self) -> &LayerEmbeddings {
        &self.layer_embeddings
    }

    /// Vectorize a sentence.
    pub fn realize(&self, sentence: &Sentence) -> Result<SentVec, Error> {
        let mut input = SentVec::with_capacity(sentence.len());

        for token in sentence {
            let form = token.form();
            let pos = token.pos().ok_or(format_err!("{}", token))?;

            input
                .tokens
                .extend_from_slice(&self.layer_embeddings.token_embeddings.embedding(form));

            input
                .tags
                .extend_from_slice(&self.layer_embeddings.tag_embeddings.embedding(pos));
        }

        Ok(input)
    }
}
