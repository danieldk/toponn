package main

import (
	"bufio"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"os"

	"github.com/danieldk/conllx"
	"github.com/danieldk/tensorflow"
	tfconfig "github.com/danieldk/tensorflow/config"
	"github.com/danieldk/toponn"
	"github.com/danieldk/toponn/cmd/common"
	"github.com/danieldk/toponn/label"
)

func mustReadGraph(config *common.TopoNNConfig) *tensorflow.Session {
	tfconf := tfconfig.ConfigProto{
		GpuOptions: &tfconfig.GPUOptions{
			PerProcessGpuMemoryFraction: config.TensorFlow.GPUMemoryFraction,
		},
	}

	opts := tensorflow.NewSessionOptions()
	defer opts.Close()
	opts.SetConfig(tfconf)

	session, err := tensorflow.NewSession(opts)
	common.ExitIfError("Could not open Tensorflow session: ", err)

	f, err := os.Open(config.TensorFlow.Graph)
	common.ExitIfError("Could not open graph: ", err)
	defer f.Close()

	data, err := ioutil.ReadAll(f)
	common.ExitIfError("Could not read graph data: ", err)

	err = session.ExtendGraph(data)
	common.ExitIfError("Could not load graph into session: ", err)

	return session
}

func main() {
	flag.Parse()

	if flag.NArg() == 0 || flag.NArg() > 3 {
		os.Exit(1)
	}

	conf := common.MustReadConfig(flag.Arg(0))
	wordEmbeds := common.MustReadEmbeddings(conf.Embeddings.Word)
	tagEmbeds := common.MustReadEmbeddings(conf.Embeddings.Tag)

	realizer := toponn.NewRealizer(wordEmbeds, tagEmbeds)
	numberer := common.MustReadLabels(conf.Fields)
	session := mustReadGraph(conf)
	defer session.Close()

	conllInput := common.FileOrStdin(flag.Args(), 1)
	defer conllInput.Close()
	conllReader := conllx.NewReader(bufio.NewReader(conllInput))

	conllOutput := common.FileOrStdout(flag.Args(), 2)
	defer conllOutput.Close()
	bufWriter := bufio.NewWriter(conllOutput)
	defer bufWriter.Flush()
	conllWriter := conllx.NewWriter(bufWriter)

	batchSize := conf.Data.BatchSize
	timesteps := conf.Data.Timesteps
	inputSize := realizer.InputLen()

	tensorBuilder := toponn.NewTensorBuilder(batchSize, timesteps, inputSize)
	batchSentences := make([]conllx.Sentence, 0, batchSize)

	for {
		sentence, err := conllReader.ReadSentence()
		if err != nil {
			if err == io.EOF {
				break
			}

			panic(err)
		}

		if len(sentence) > conf.Data.Timesteps {
			// FIXME: we need a better implementation of this! But for now,
			// just skip them.
			continue
		}

		// Copy, because the reader recycles sentences.
		sentenceCopy := make([]conllx.Token, len(sentence))
		copy(sentenceCopy, sentence)

		batchSentences = append(batchSentences, sentenceCopy)
		tensorBuilder.Add(realizer.Realize(sentenceCopy))

		if len(batchSentences) == batchSize {
			labelBatch(session, numberer, batchSentences, tensorBuilder.Tensor())
			writeSentences(batchSentences, conllWriter)

			batchSentences = make([]conllx.Sentence, 0)
		}
	}

	if len(batchSentences) > 0 {
		labelBatch(session, numberer, batchSentences, tensorBuilder.Tensor())
		writeSentences(batchSentences, conllWriter)
	}
}

func labelBatch(session *tensorflow.Session, numberer *label.LabelNumberer, batchSentences []conllx.Sentence, batch *tensorflow.Float32Tensor) {
	outputs, err := session.Run(map[string]tensorflow.Tensor{"model_2/inputs": batch}, []string{"model_2/predictions"})
	common.ExitIfError("Error running graph: ", err)
	predictions := outputs["model_2/predictions"].(*tensorflow.Int32Tensor)

	for idx, sent := range batchSentences {
		sentencePredictions := predictions.Get([]int{idx})
		for tokenIdx := 0; tokenIdx < common.MinInt(len(sentencePredictions), len(sent)); tokenIdx++ {
			if label, ok := numberer.Label(int(sentencePredictions[tokenIdx])); ok {
				sent[tokenIdx].SetFeatures(map[string]string{"tf": label})
			} else {
				log.Printf("Impossible label predicted (0 is padding): %d\n", sentencePredictions[tokenIdx])
			}
		}
	}
}

func writeSentences(sentences []conllx.Sentence, writer *conllx.Writer) {
	for _, sent := range sentences {
		writer.WriteSentence(sent)
	}
}
