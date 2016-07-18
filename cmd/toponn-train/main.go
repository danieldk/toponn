package main

import (
	"flag"
	"log"
	"os"

	"github.com/danieldk/conllx"
	"github.com/danieldk/toponn"
	"github.com/danieldk/toponn/cmd/common"
	"github.com/danieldk/toponn/label"
	"github.com/sbinet/go-hdf5"
)

func main() {
	flag.Parse()

	if flag.NArg() != 3 {
		os.Exit(1)
	}

	conf := common.MustReadConfig(flag.Arg(0))
	wordEmbeds := common.MustReadEmbeddings(conf.Embeddings.Word)
	tagEmbeds := common.MustReadEmbeddings(conf.Embeddings.Tag)

	trainFile, err := os.Open(flag.Arg(1))
	common.ExitIfError("Cannot open file with training data: ", err)
	defer trainFile.Close()

	var labelNumberer *label.LabelNumberer
	if conf.Fields != "" {
		if _, err := os.Stat(conf.Fields); err == nil {
			log.Printf("Transitions filename %s exists, reusing...", conf.Fields)
			labelNumberer = common.MustReadLabels(conf.Fields)
		}
	}

	var realizer *toponn.TrainRealizer
	if labelNumberer == nil {
		realizer = toponn.NewTrainRealizer(wordEmbeds, tagEmbeds)
	} else {
		realizer = toponn.NewTrainRealizerWithLabelNumberer(wordEmbeds, tagEmbeds, labelNumberer)
	}

	outputFile, err := hdf5.CreateFile(flag.Arg(2), hdf5.F_ACC_TRUNC)
	common.ExitIfError("Error creating file: ", err)
	defer outputFile.Close()

	writer := toponn.NewBatchWriter(outputFile, conf.Data.BatchSize, conf.Data.Timesteps)
	defer writer.Close()

	err = common.ProcessData(trainFile, func(sentence []conllx.Token) error {
		if instance, err := realizer.Realize(sentence); err == nil {
			return writer.Write(instance)
		} else {
			return err
		}
	})
	common.ExitIfError("Error processing data: ", err)

	if conf.Fields != "" {
		if _, err := os.Stat(conf.Fields); err != nil {
			mustWriteLabels(realizer.LabelNumberer(), conf.Fields)
		}
	}
}

func mustWriteLabels(l label.LabelNumberer, filename string) {
	f, err := os.Create(filename)
	common.ExitIfError("Could not create label file: ", err)
	defer f.Close()

	err = l.WriteLabelNumberer(f)
	common.ExitIfError("Could not write labels: ", err)
}
