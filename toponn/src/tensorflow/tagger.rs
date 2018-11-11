use std::cmp::min;
use std::io::Read;

use conllx::Sentence;
use protobuf::Message;
use tf::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

use super::tensor::TensorBuilder;
use tf_proto::ConfigProto;
use {ErrorKind, Numberer, Result, SentVectorizer, Tag};

const INITIAL_SEQUENCE_LENGTH: usize = 100;

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Model {
    /// Model batch size, should be kept constant between training and
    /// prediction.
    pub batch_size: usize,

    /// The filename of the frozen Tensorflow graph.
    pub filename: String,

    /// Operation names for the frozen tensorflow graph.
    pub op_names: OpNames,

    /// Thread pool size for parallel processing within a computation
    /// graph op.
    pub intra_op_parallelism_threads: usize,

    /// Thread pool size for parallel processing of independent computation
    /// graph ops.
    pub inter_op_parallelism_threads: usize,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpNames {
    pub token_embeds_op: String,
    pub tag_embeds_op: String,

    pub tokens_op: String,
    pub tags_op: String,
    pub seq_lens_op: String,

    pub predicted_op: String,
}

pub struct Tagger {
    session: Session,

    vectorizer: SentVectorizer,
    labels: Numberer<String>,
    builder: TensorBuilder,

    token_embeds_op: Operation,
    tokens_op: Operation,
    tags_op: Operation,
    seq_lens_op: Operation,
    tag_embeds_op: Operation,

    predicted_op: Operation,
}

impl Tagger {
    pub fn load_graph<R>(
        mut r: R,
        vectorizer: SentVectorizer,
        labels: Numberer<String>,
        model: &Model,
    ) -> Result<Self>
    where
        R: Read,
    {
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        let opts = ImportGraphDefOptions::new();
        let mut graph = Graph::new();
        graph.import_graph_def(&data, &opts)?;

        let mut session_opts = SessionOptions::new();
        session_opts.set_config(&tf_model_to_protobuf(&model)?)?;
        let session = Session::new(&session_opts, &graph)?;

        let op_names = &model.op_names;

        let token_embeds_op = graph.operation_by_name_required(&op_names.token_embeds_op)?;
        let tag_embeds_op = graph.operation_by_name_required(&op_names.tag_embeds_op)?;

        let tokens_op = graph.operation_by_name_required(&op_names.tokens_op)?;
        let tags_op = graph.operation_by_name_required(&op_names.tags_op)?;
        let seq_lens_op = graph.operation_by_name_required(&op_names.seq_lens_op)?;

        let predicted_op = graph.operation_by_name_required(&op_names.predicted_op)?;

        Ok(Tagger {
            session,

            vectorizer,
            labels,
            builder: TensorBuilder::new(model.batch_size, INITIAL_SEQUENCE_LENGTH),

            tokens_op,
            tags_op,
            seq_lens_op,
            token_embeds_op,
            tag_embeds_op,
            predicted_op,
        })
    }

    fn ensure_builder_seq_len(&mut self, len: usize) {
        if self.builder.max_seq_len() >= len {
            return;
        }

        self.builder = TensorBuilder::new(self.builder.batch_size(), len);
    }

    fn tag_sequences(&mut self) -> Result<Tensor<i32>> {
        let mut run_args = SessionRunArgs::new();

        let embeds = self.vectorizer.layer_embeddings();

        // Embedding inputs
        run_args.add_feed(&self.token_embeds_op, 0, embeds.token_embeddings().data());
        run_args.add_feed(&self.tag_embeds_op, 0, embeds.tag_embeddings().data());

        // Sequence inputs
        run_args.add_feed(&self.seq_lens_op, 0, self.builder.seq_lens());
        run_args.add_feed(&self.tokens_op, 0, self.builder.tokens());
        run_args.add_feed(&self.tags_op, 0, self.builder.tags());

        let predictions_token = run_args.request_fetch(&self.predicted_op, 0);

        self.session.run(&mut run_args)?;

        Ok(run_args.fetch(predictions_token)?)
    }
}

impl Tag for Tagger {
    fn tag_sentences(&mut self, sentences: &[Sentence]) -> Result<Vec<Vec<&str>>> {
        if sentences.len() > self.builder.batch_size() {
            bail!(ErrorKind::IncorrectBatchSize(
                sentences.len(),
                self.builder.batch_size(),
            ));
        }

        // Find maximum sentence size.
        let max_seq_len = sentences
            .iter()
            .map(|s| s.as_tokens().len())
            .max()
            .unwrap_or(0);
        self.ensure_builder_seq_len(max_seq_len);

        self.builder.clear();

        // Fill the batch.
        for sentence in sentences {
            let input = self.vectorizer.realize(sentence)?;
            self.builder.add(&input);
        }

        // Tag the batch
        let tag_tensor = self.tag_sequences()?;

        // Convert label numbers to labels.
        let numberer = &self.labels;
        let mut labels = Vec::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let seq_len = min(tag_tensor.dims()[1] as usize, sentence.as_tokens().len());
            let offset = idx * tag_tensor.dims()[1] as usize;
            let seq = &tag_tensor[offset..offset + seq_len];

            labels.push(
                seq.iter()
                    .map(|&label| numberer.value(label as usize).unwrap().as_str())
                    .collect(),
            );
        }

        Ok(labels)
    }
}

fn tf_model_to_protobuf(model: &Model) -> Result<Vec<u8>> {
    let mut config_proto = ConfigProto::new();
    config_proto.intra_op_parallelism_threads = model.intra_op_parallelism_threads as i32;
    config_proto.inter_op_parallelism_threads = model.inter_op_parallelism_threads as i32;

    let mut bytes = Vec::new();
    config_proto.write_to_vec(&mut bytes)?;

    Ok(bytes)
}
