use std::cmp::min;
use std::io::Read;

use conllx::Sentence;
use protobuf::Message;
use tf::{Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, StepWithGraph, Tensor};

use {ErrorKind, Numberer, Result, SentVectorizer, Tag};
use super::tensor::TensorBuilder;
use tf_proto::ConfigProto;

const INITIAL_SEQUENCE_LENGTH: usize = 100;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelConfig {
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

impl ModelConfig {
    pub fn new_session(&self, model: &Model) -> Result<Session> {
        let mut session_opts = SessionOptions::new();
        session_opts.set_config(&tf_model_to_protobuf(&self)?)?;
        Ok(Session::new(&session_opts, &model.graph)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpNames {
    pub token_embeds_op: String,
    pub tag_embeds_op: String,

    pub tokens_op: String,
    pub tags_op: String,
    pub seq_lens_op: String,

    pub predicted_op: String,
}

pub struct Model {
    vectorizer: SentVectorizer,
    labels: Numberer<String>,

    graph: Graph,

    token_embeds_op: Operation,
    tokens_op: Operation,
    tags_op: Operation,
    seq_lens_op: Operation,
    tag_embeds_op: Operation,

    predicted_op: Operation,
}

impl Model {
    pub fn load_graph<R>(
        mut r: R,
        vectorizer: SentVectorizer,
        labels: Numberer<String>,
        model: &ModelConfig,
    ) -> Result<Self>
    where
        R: Read,
    {
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        let opts = ImportGraphDefOptions::new();
        let mut graph = Graph::new();
        graph.import_graph_def(&data, &opts)?;

        let op_names = &model.op_names;

        let token_embeds_op = graph.operation_by_name_required(&op_names.token_embeds_op)?;
        let tag_embeds_op = graph.operation_by_name_required(&op_names.tag_embeds_op)?;

        let tokens_op = graph.operation_by_name_required(&op_names.tokens_op)?;
        let tags_op = graph.operation_by_name_required(&op_names.tags_op)?;
        let seq_lens_op = graph.operation_by_name_required(&op_names.seq_lens_op)?;

        let predicted_op = graph.operation_by_name_required(&op_names.predicted_op)?;

        Ok(Model {
            vectorizer,
            labels,

            graph,

            tokens_op,
            tags_op,
            seq_lens_op,
            token_embeds_op,
            tag_embeds_op,
            predicted_op,
        })
    }
}

pub struct Tagger<'a> {
    model: &'a Model,
    builder: TensorBuilder,
    session: &'a mut Session,
}

impl<'a> Tagger<'a> {
    pub fn new(model_config: &ModelConfig, model: &'a Model, session: &'a mut Session) -> Self {
        Tagger {
            model,
            builder: TensorBuilder::new(model_config.batch_size, INITIAL_SEQUENCE_LENGTH),
            session,
        }
    }

    fn ensure_builder_seq_len(&mut self, len: usize) {
        if self.builder.max_seq_len() >= len {
            return;
        }

        self.builder = TensorBuilder::new(self.builder.batch_size(), len);
    }

    fn tag_sequences(&mut self) -> Result<Tensor<i32>> {
        let mut step = StepWithGraph::new();

        let embeds = self.model.vectorizer.layer_embeddings();

        // Embedding inputs
        step.add_input(
            &self.model.token_embeds_op,
            0,
            embeds.token_embeddings().data(),
        );
        step.add_input(&self.model.tag_embeds_op, 0, embeds.tag_embeddings().data());

        // Sequence inputs
        step.add_input(&self.model.seq_lens_op, 0, self.builder.seq_lens());
        step.add_input(&self.model.tokens_op, 0, self.builder.tokens());
        step.add_input(&self.model.tags_op, 0, self.builder.tags());

        let predictions_token = step.request_output(&self.model.predicted_op, 0);

        self.session.run(&mut step)?;

        Ok(step.take_output(predictions_token)?)
    }
}

impl<'a> Tag for Tagger<'a> {
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
            let input = self.model.vectorizer.realize(sentence)?;
            self.builder.add(&input);
        }

        // Tag the batch
        let tag_tensor = self.tag_sequences()?;

        // Convert label numbers to labels.
        let numberer = &self.model.labels;
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

fn tf_model_to_protobuf(model: &ModelConfig) -> Result<Vec<u8>> {
    let mut config_proto = ConfigProto::new();
    config_proto.intra_op_parallelism_threads = model.intra_op_parallelism_threads as i32;
    config_proto.inter_op_parallelism_threads = model.inter_op_parallelism_threads as i32;

    let mut bytes = Vec::new();
    config_proto.write_to_vec(&mut bytes)?;

    Ok(bytes)
}
