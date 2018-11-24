use conllx::Sentence;

use failure::Error;
use {Numberer, SentVectorizer};

/// Data types collects (and typically stores) vectorized sentences.
pub trait Collector {
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error>;

    fn vectorizer(&self) -> &SentVectorizer;
}

/// Collector that does not store the vectorized sentences.
///
/// This collector can be used to construct lookup tables as a
/// side-effect of vectorizing the input.
pub struct NoopCollector {
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
}

impl NoopCollector {
    pub fn new(numberer: Numberer<String>, vectorizer: SentVectorizer) -> NoopCollector {
        NoopCollector {
            numberer,
            vectorizer,
        }
    }

    pub fn labels(&self) -> &Numberer<String> {
        &self.numberer
    }
}

impl Collector for NoopCollector {
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error> {
        self.vectorizer.realize(sentence)?;

        for token in sentence {
            let features = token
                .features()
                .ok_or(format_err!(
                    "No features field with a topological field (tf) feature: {}",
                    token
                ))?.as_map();
            let opt_tf = features.get("tf").ok_or(format_err!(
                "No features field with a topological field (tf) feature: {}",
                token
            ))?;
            let tf = opt_tf.clone().ok_or(format_err!(
                "Topological field feature (tf) without a value: {}",
                token
            ))?;

            self.numberer.add(tf.to_owned());
        }

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
