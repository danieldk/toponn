package toponn

import (
	"fmt"
	"reflect"

	"github.com/sbinet/go-hdf5"
)

var _ Writer = new(HDF5Writer)

type HDF5Writer struct {
	f         *hdf5.File
	batchSize int
	timesteps int
	inputSize int
	sequence  int // Current sequence within the batch.
	batch     int // Current batch.
	data      []float32
	labels    []int32
}

func NewBatchWriter(f *hdf5.File, batchSize, timesteps int) *HDF5Writer {
	return &HDF5Writer{
		f:         f,
		batchSize: batchSize,
		timesteps: timesteps,
		labels:    make([]int32, batchSize*timesteps),
	}
}

func (bw *HDF5Writer) Close() error {
	return bw.writeBatch()
}

func (bw *HDF5Writer) Write(instance TrainInstance) error {
	input := instance.input

	if bw.data == nil {
		bw.inputSize = len(input.data) / input.timesteps
		bw.data = make([]float32, bw.batchSize*bw.timesteps*bw.inputSize)
	}

	if bw.sequence >= bw.batchSize {
		bw.writeBatch()
		bw.clearBatch()
	}

	copy(bw.data[bw.sequence*bw.timesteps*bw.inputSize:],
		input.data[:min(len(input.data), bw.timesteps*bw.inputSize)])
	copy(bw.labels[bw.sequence*bw.timesteps:],
		instance.labels[:min(len(instance.labels), bw.batchSize)])

	bw.sequence++

	return nil
}

func (bw *HDF5Writer) clearBatch() {
	bw.sequence = 0
	bw.batch++
	for idx := range bw.data {
		bw.data[idx] = 0
	}

	for idx := range bw.labels {
		bw.labels[idx] = 0
	}
}

func (bw *HDF5Writer) writeBatch() error {
	group, err := bw.f.CreateGroup(fmt.Sprintf("batch%d", bw.batch))
	if err != nil {
		return err
	}
	defer group.Close()

	if err := bw.writeData(group); err != nil {
		return err
	}

	if err := bw.writeLabels(group); err != nil {
		return err
	}

	return nil
}

func (bw *HDF5Writer) writeData(g *hdf5.Group) error {
	dims := []uint{uint(bw.batchSize), uint(bw.timesteps), uint(bw.inputSize)}
	return writeData(g, "inputs", dims, bw.data)
}

func (bw *HDF5Writer) writeLabels(g *hdf5.Group) error {
	dims := []uint{uint(bw.batchSize), uint(bw.timesteps)}
	return writeData(g, "labels", dims, bw.labels)
}

func writeData(f *hdf5.Group, name string, dims []uint,
	data interface{}) error {
	space, err := hdf5.CreateSimpleDataspace(dims, nil)
	if err != nil {
		return err
	}

	first := reflect.ValueOf(data).Index(0)

	dtype, err := hdf5.NewDatatypeFromValue(first.Interface())
	if err != nil {
		return err
	}

	dset, err := f.CreateDataset(name, dtype, space)
	if err != nil {
		return err
	}
	defer dset.Close()

	err = dset.Write(first.Addr().Interface())
	if err != nil {
		return err
	}

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}
