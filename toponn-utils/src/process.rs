use std::collections::BTreeMap;
use std::io::Write;
use std::mem;

use conllx;
use conllx::{Features, Sentence, WriteSentence};
use toponn::Tag;

use error::*;

// Wrap the sentence processing in a data type. This has the benefit that
// we can use a destructor to write the last (possibly incomplete) batch.
pub struct SentProcessor<'a, W>
where
    W: Write,
{
    tagger: &'a mut Tag,
    writer: conllx::Writer<W>,
    batch_size: usize,
    batch_sents: Vec<Sentence>,
}

impl<'a, W> SentProcessor<'a, W>
where
    W: Write,
{
    pub fn new(tagger: &'a mut Tag, writer: conllx::Writer<W>, batch_size: usize) -> Self {
        SentProcessor {
            tagger,
            writer,
            batch_size,
            batch_sents: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.batch_sents.clear();
    }

    pub fn process(&mut self, sent: Sentence) -> Result<()> {
        self.batch_sents.push(sent);

        if self.batch_sents.len() == self.batch_size {
            let mut batch = Vec::with_capacity(self.batch_size);
            mem::swap(&mut batch, &mut self.batch_sents);

            let labels = labels_to_owned(self.tagger.tag_sentences(&batch)?);
            self.write_sent_labels(labels)?;
        }

        Ok(())
    }

    fn write_sent_labels(&mut self, labels: Vec<Vec<String>>) -> Result<()>
    where
        W: Write,
    {
        for (sentence, sent_labels) in self.batch_sents.iter_mut().zip(labels.iter()) {
            {
                let mut tokens = sentence.as_tokens_mut();

                for i in 0..tokens.len() {
                    // Obtain the feature mapping or construct a fresh one.
                    let mut features = tokens[i]
                        .features()
                        .map(Features::as_map)
                        .cloned()
                        .unwrap_or(BTreeMap::new());

                    // Insert the topological field.
                    features.insert(String::from("tf"), Some(sent_labels[i].to_owned()));

                    tokens[i].set_features(Some(Features::from_iter(features)));
                }
            }

            self.writer.write_sentence(&sentence)?;
        }

        Ok(())
    }
}

impl<'a, W> Drop for SentProcessor<'a, W>
where
    W: Write,
{
    fn drop(&mut self) {
        if !self.batch_sents.is_empty() {
            let labels = labels_to_owned(match self.tagger.tag_sentences(&self.batch_sents) {
                Ok(labels) => labels,
                Err(err) => {
                    eprintln!("Error tagging sentences: {}", err);
                    return;
                }
            });

            if let Err(err) = self.write_sent_labels(labels) {
                eprintln!("Error writing sentences: {}", err);
            }
        }
    }
}

fn labels_to_owned(labels: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    labels
        .into_iter()
        .map(|sv| sv.into_iter().map(str::to_owned).collect())
        .collect()
}
