use std::cmp::min;
use std::path::Path;

use conllx::Sentence;
use hdf5;
use hdf5::IntoData;
use itertools::multizip;

use {ErrorKind, Numberer, Result, SentVec, SentVectorizer};

/// Data types collects (and typically stores) vectorized sentences.
pub trait Collector {
    fn collect(&mut self, sentence: &Sentence) -> Result<()>;
}

/// Collector that stores vectorized sentences in a HDF5 container.
pub struct HDF5Collector {
    writer: HDF5Writer,
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
}

impl HDF5Collector {
    pub fn new<P>(
        hdf5_path: P,
        numberer: Numberer<String>,
        vectorizer: SentVectorizer,
        batch_size: usize,
    ) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Ok(HDF5Collector {
            writer: HDF5Writer::new(hdf5::File::new(hdf5_path)?, batch_size),
            numberer: numberer,
            vectorizer: vectorizer,
        })
    }

    pub fn labels(&self) -> &Numberer<String> {
        &self.numberer
    }
}

impl Collector for HDF5Collector {
    fn collect(&mut self, sentence: &Sentence) -> Result<()> {
        let input = self.vectorizer.realize(sentence)?;

        let mut labels = Vec::with_capacity(sentence.as_tokens().len());
        for token in sentence {
            let features = token
                .features()
                .ok_or(ErrorKind::MissingTopologicalField(format!("{}", token)))?
                .as_map();
            let opt_tf = features
                .get("tf")
                .ok_or(ErrorKind::MissingTopologicalField(format!("{}", token)))?;
            let tf = opt_tf
                .clone()
                .ok_or(ErrorKind::MissingTopologicalField(format!("{}", token)))?;

            labels.push(self.numberer.add(tf.to_owned()));
        }

        self.writer.write(&labels, input)?;

        Ok(())
    }
}

pub struct HDF5Writer {
    file: hdf5::File,
    batch: usize,
    batch_size: usize,
    batch_idx: usize,
    labels: Vec<Vec<i32>>,
    tokens: Vec<Vec<i32>>,
    tags: Vec<Vec<i32>>,
    lens: Vec<i32>,
}

impl HDF5Writer {
    pub fn new(hdf5_file: hdf5::File, batch_size: usize) -> Self {
        HDF5Writer {
            file: hdf5_file,
            batch: 0,
            batch_size,
            batch_idx: 0,
            labels: Vec::new(),
            tokens: Vec::new(),
            tags: Vec::new(),
            lens: Vec::new(),
        }
    }

    pub fn write(&mut self, labels: &[usize], input: SentVec) -> Result<()> {
        let (tokens, tags) = input.to_parts();

        self.tags.push(tags);
        self.tokens.push(tokens);
        self.labels.push(labels.iter().map(|l| *l as i32).collect());
        self.lens.push(labels.len() as i32);

        self.batch_idx += 1;

        if self.batch_idx >= self.batch_size {
            self.write_batch()?;
            self.clear_batch();
        }

        Ok(())
    }

    fn clear_batch(&mut self) {
        self.batch_idx = 0;
        self.batch += 1;

        self.tokens.clear();
        self.tags.clear();
        self.labels.clear();
        self.lens.clear();
    }

    fn write_batch(&mut self) -> Result<()> {
        let time_steps: usize = self.tokens
            .iter()
            .map(Vec::len)
            .max()
            .expect("Attempting to write empty batch");

        let mut tokens_batch = vec![0; self.batch_size * time_steps];
        let mut tags_batch = vec![0; self.batch_size * time_steps];
        let mut labels_batch = vec![0; self.batch_size * time_steps];

        for (idx, tokens, tags, labels) in
            multizip((0..self.batch_size, &self.tokens, &self.tags, &self.labels))
        {
            let offset = idx * time_steps;
            let len = min(time_steps, tokens.len());
            tokens_batch[offset..offset + len].copy_from_slice(&tokens[..len]);
            tags_batch[offset..offset + len].copy_from_slice(&tags[..len]);
            labels_batch[offset..offset + len].copy_from_slice(&labels[..len]);
        }

        self.write_batch_raw("tokens", &tokens_batch)?;
        self.write_batch_raw("tags", &tags_batch)?;
        self.write_batch_raw("labels", &labels_batch)?;

        let mut lens_batch = vec![0; self.batch_size];
        lens_batch[0..self.lens.len()].copy_from_slice(&self.lens);

        let mut writer = hdf5::Writer::new(
            &self.file,
            &format!("batch{}-lens", self.batch),
            &[self.batch_size],
        );

        writer.write(lens_batch.into_data()?, &[0], &[self.batch_size])?;

        Ok(())
    }

    fn write_batch_raw(&self, layer: &str, data: &[i32]) -> Result<()> {
        let mut writer = hdf5::Writer::new(
            &self.file,
            &format!("batch{}-{}", self.batch, layer),
            &[self.batch_size, data.len() / self.batch_size],
        );

        writer.write(
            data.into_data()?,
            &[0, 0],
            &[self.batch_size, data.len() / self.batch_size],
        )?;
        Ok(())
    }
}

impl Drop for HDF5Writer {
    fn drop(&mut self) {
        if self.batch_idx > 0 {
            self.write_batch().expect("Cannot write last batch");
            self.file
                .write("batches", self.batch + 1)
                .expect("Cannot write last batch");
        }
    }
}
