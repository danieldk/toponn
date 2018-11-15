use std::cmp::min;

use tf::Tensor;

use SentVec;

pub struct TensorBuilder {
    sequence: usize,
    sequence_lens: Tensor<i32>,
    tokens: Tensor<i32>,
    tags: Tensor<i32>,
}

impl TensorBuilder {
    pub fn new(batch_size: usize, time_steps: usize) -> Self {
        TensorBuilder {
            sequence: 0,
            sequence_lens: Tensor::new(&[batch_size as u64]),
            tokens: Tensor::new(&[batch_size as u64, time_steps as u64]),
            tags: Tensor::new(&[batch_size as u64, time_steps as u64]),
        }
    }

    pub fn add(&mut self, input: &SentVec) {
        assert!((self.sequence as u64) < self.tokens.dims()[0]);

        let max_seq_len = self.tokens.dims()[1] as usize;

        // Number of time steps to copy.
        let timesteps = min(max_seq_len, input.tokens.len());
        self.sequence_lens[self.sequence] = timesteps as i32;

        let offset = self.sequence * max_seq_len;

        let token_seq = &mut self.tokens[offset..offset + timesteps];
        token_seq.copy_from_slice(&input.tokens[..timesteps]);

        let tag_seq = &mut self.tags[offset..offset + timesteps];
        tag_seq.copy_from_slice(&input.tags[..timesteps]);

        self.sequence += 1;
    }

    pub fn seq_lens(&self) -> &Tensor<i32> {
        &self.sequence_lens
    }

    pub fn tags(&self) -> &Tensor<i32> {
        &self.tags
    }

    pub fn tokens(&self) -> &Tensor<i32> {
        &self.tokens
    }
}
