package toponn

import "github.com/danieldk/tensorflow"

// A TensorBuilder constructs a tensor of the given size and populates
// it with the inputs that are provided.
type TensorBuilder struct {
	batchSize int
	timesteps int
	inputSize int
	sequence  int
	tensor    *tensorflow.Float32Tensor
}

// Create a new TensorBuilder for a tensor of the given batch size, time steps,
// and input size.
func NewTensorBuilder(batchSize, timesteps, inputSize int) *TensorBuilder {
	return &TensorBuilder{
		batchSize: batchSize,
		timesteps: timesteps,
		inputSize: inputSize,
	}
}

// Add an input to the tensor. If as many inputs as the batch size were added,
// writing starts again at the beginning. The batchSize-th input overwrites
// the 0-th input. This is an intentional optimization, because it allows reuse
// of the TensorBuilder without reallocation.
func (b *TensorBuilder) Add(input Input) {
	if b.tensor == nil {
		b.tensor = tensorflow.NewFloat32Tensor([]int{b.batchSize, b.timesteps, b.inputSize})
	}

	paddedInput := padInput(&input, b.timesteps)

	b.tensor.Assign([]int{b.sequence}, paddedInput)

	b.sequence++
	if b.sequence == b.batchSize {
		b.sequence = 0
	}
}

// Get the Tensor.
func (w *TensorBuilder) Tensor() *tensorflow.Float32Tensor {
	return w.tensor
}

func padInput(input *Input, sequenceLen int) []float32 {
	inputSize := len(input.Data()) / input.Timesteps()
	padded := make([]float32, inputSize*sequenceLen)
	copy(padded, input.Data())
	return padded
}
