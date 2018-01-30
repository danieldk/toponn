use std::collections::HashMap;

use conllx::Sentence;
use tf_embed::Embeddings;

use {ErrorKind, Result};

/// Sentence represented as a vector.
///
/// This data type represents a sentence as vectors (`Vec`) of tokens and
/// part-of-speech indices. Such a vector is typically the input to a
/// sequence labeling graph.
pub struct SentVec {
    pub tokens: Vec<i32>,
    pub tags: Vec<i32>,
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

    /// Decompose the sentence vector into vectors of token indices and
    /// part-of-speech tag indices.
    pub fn to_parts(self) -> (Vec<i32>, Vec<i32>) {
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
    pub fn realize(&self, sentence: &Sentence) -> Result<SentVec> {
        let mut input = SentVec::with_capacity(sentence.as_tokens().len());

        for token in sentence {
            let form = token.form();
            let pos = token
                .pos()
                .ok_or(ErrorKind::MissingPOSTag(format!("{}", token)))?;

            input.tokens.push(lookup_value_or_unknown(
                self.layer_embeddings.token_embeddings.indices(),
                form,
            ));

            input.tags.push(lookup_value_or_unknown(
                self.layer_embeddings.tag_embeddings.indices(),
                pos,
            ));
        }

        Ok(input)
    }
}

fn lookup_value_or_unknown(m: &HashMap<String, usize>, value: &str) -> i32 {
    if let Some(idx) = m.get(value) {
        *idx as i32
    } else {
        lookup_unknown(m)
    }
}

fn lookup_unknown(m: &HashMap<String, usize>) -> i32 {
    m.get("<UNKNOWN-TOKEN>").cloned().unwrap_or(0) as i32
}
