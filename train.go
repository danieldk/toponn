package toponn

import (
	"errors"
	"strconv"

	"github.com/danieldk/conllx"
	"github.com/danieldk/go2vec"
	"github.com/danieldk/toponn/label"
)

type TrainInstance struct {
	input  Input
	labels []int32
}

type TrainRealizer struct {
	labelNumberer label.LabelNumberer
	realizer      *Realizer
}

func NewTrainRealizer(wordEmbeds, tagEmbeds *go2vec.Embeddings) *TrainRealizer {
	return &TrainRealizer{
		labelNumberer: label.NewLabelNumberer(),
		realizer:      NewRealizer(wordEmbeds, tagEmbeds),
	}
}

func NewTrainRealizerWithLabelNumberer(wordEmbeds, tagEmbeds *go2vec.Embeddings, l *label.LabelNumberer) *TrainRealizer {
	return &TrainRealizer{
		labelNumberer: *l,
		realizer:      NewRealizer(wordEmbeds, tagEmbeds),
	}
}

func (ivr *TrainRealizer) LabelNumberer() label.LabelNumberer {
	return ivr.labelNumberer
}

func (ivr *TrainRealizer) Realize(sentence []conllx.Token) (TrainInstance, error) {
	input := ivr.realizer.Realize(sentence)

	labels := make([]int32, len(sentence))
	depths := make([]int32, len(sentence))
	for idx, token := range sentence {
		features, ok := token.Features()
		if !ok {
			return TrainInstance{}, errors.New("Token without features.")
		}

		field, ok := features.FeaturesMap()["tf"]
		if !ok {
			return TrainInstance{}, errors.New("No topological field feature.")
		}
		labels[idx] = int32(ivr.labelNumberer.Number(field))

		depthVal, ok := features.FeaturesMap()["tfdepth"]
		if !ok {
			return TrainInstance{}, errors.New("No topological field depth feature.")
		}

		depth, err := strconv.ParseInt(depthVal, 10, 0)
		if err != nil {
			return TrainInstance{}, err
		}

		depths[idx] = int32(depth)
	}

	return TrainInstance{
		input:  input,
		labels: labels,
	}, nil
}
