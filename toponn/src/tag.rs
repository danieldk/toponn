use conllx::Sentence;

use Result;

/// Trait for topological field taggers.
pub trait Tag {
    fn tag_sentences(&mut self, sentences: &[Sentence]) -> Result<Vec<Vec<&str>>>;
}
