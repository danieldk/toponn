package toponn

type Writer interface {
	Close() error
	Write(instance TrainInstance) error
}
