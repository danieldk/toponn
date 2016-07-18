package common

import (
	"bufio"
	"log"
	"os"

	"gopkg.in/cheggaaa/pb.v1"

	"github.com/danieldk/conllx"
	"github.com/danieldk/go2vec"
	"github.com/danieldk/toponn/label"
)

func CreateFileProgress(f *os.File) (*pb.ProgressBar, error) {
	if fi, err := f.Stat(); err == nil {
		bar := pb.New64(fi.Size())
		bar.SetUnits(pb.U_BYTES)
		return bar, nil
	} else {
		return nil, err
	}
}

func ExitIfError(prefix string, err error) {
	if err != nil {
		log.Fatal(prefix, err.Error())
	}
}

type DataFun func([]conllx.Token) error

func ProcessData(f *os.File, fun DataFun) error {
	bar, err := CreateFileProgress(f)
	if err != nil {
		return err
	}
	defer bar.Finish()
	bar.Start()

	r := conllx.NewReader(bufio.NewReader(bar.NewProxyReader(f)))

	for {
		s, err := r.ReadSentence()
		if err != nil {
			break
		}

		if err := fun(s); err != nil {
			return err
		}
	}

	return nil
}

func MustReadEmbeddings(config Embedding) *go2vec.Embeddings {
	if config.Filename == "" {
		return nil
	}

	if config.Normalize {
		log.Printf("Reading normalized vectors from %s...\n", config.Filename)
	} else {
		log.Printf("Reading vectors from %s...\n", config.Filename)
	}

	vecf, err := os.Open(config.Filename)
	ExitIfError("Cannot open vectors file: ", err)
	defer vecf.Close()

	vecs, err := go2vec.ReadWord2VecBinary(bufio.NewReader(vecf), config.Normalize)
	ExitIfError("Cannot read vectors: ", err)

	return vecs
}

func MustReadLabels(filename string) *label.LabelNumberer {
	f, err := os.Open(filename)
	ExitIfError("Could not open label file: ", err)
	defer f.Close()

	l := label.NewLabelNumberer()
	err = l.Read(f)
	ExitIfError("Could not read label file: ", err)

	return &l
}
