use std::cmp::min;

use tf::Tensor;

use SentVec;

pub struct TensorBuilder {
    sequence: usize,
    sequence_lens: Tensor<i32>,
    tokens: Tensor<f32>,
    tags: Tensor<f32>,
}

impl TensorBuilder {
    pub fn new(
        batch_size: usize,
        time_steps: usize,
        token_embed_size: usize,
        tag_embed_size: usize,
    ) -> Self {
        TensorBuilder {
            sequence: 0,
            sequence_lens: Tensor::new(&[batch_size as u64]),
            tokens: Tensor::new(&[
                batch_size as u64,
                time_steps as u64,
                token_embed_size as u64,
            ]),
            tags: Tensor::new(&[batch_size as u64, time_steps as u64, tag_embed_size as u64]),
        }
    }

    pub fn add(&mut self, input: &SentVec) {
        assert!((self.sequence as u64) < self.tokens.dims()[0]);

        let max_seq_len = self.tokens.dims()[1] as usize;
        let token_embed_size = self.tokens.dims()[2] as usize;
        let tag_embed_size = self.tags.dims()[2] as usize;

        // Number of time steps to copy.
        let timesteps = min(max_seq_len, input.tokens.len() / token_embed_size);
        self.sequence_lens[self.sequence] = timesteps as i32;

        let token_offset = self.sequence * max_seq_len * token_embed_size;
        let token_seq =
            &mut self.tokens[token_offset..token_offset + (token_embed_size * timesteps)];
        token_seq.copy_from_slice(&input.tokens[..token_embed_size * timesteps]);

        let tag_offset = self.sequence * max_seq_len * tag_embed_size;
        let tag_seq = &mut self.tags[tag_offset..tag_offset + (tag_embed_size * timesteps)];
        tag_seq.copy_from_slice(&input.tags[..tag_embed_size * timesteps]);

        self.sequence += 1;
    }

    pub fn seq_lens(&self) -> &Tensor<i32> {
        &self.sequence_lens
    }

    pub fn tags(&self) -> &Tensor<f32> {
        &self.tags
    }

    pub fn tokens(&self) -> &Tensor<f32> {
        &self.tokens
    }
}
