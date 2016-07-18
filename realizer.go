package toponn

import (
	"github.com/danieldk/conllx"
	"github.com/danieldk/go2vec"
)

type Input struct {
	data      []float32
	timesteps int
}

func (i *Input) Data() []float32 {
	return i.data
}

func (i *Input) Timesteps() int {
	return i.timesteps
}

type Realizer struct {
	wordEmbeds *go2vec.Embeddings
	tagEmbeds  *go2vec.Embeddings
}

func NewRealizer(wordEmbeds, tagEmbeds *go2vec.Embeddings) *Realizer {
	return &Realizer{
		wordEmbeds: wordEmbeds,
		tagEmbeds:  tagEmbeds,
	}
}

func (ivr *Realizer) Realize(sentence []conllx.Token) Input {
	embeddingsSize := ivr.InputLen()

	data := make([]float32, len(sentence)*embeddingsSize)

	for idx, token := range sentence {
		wordEmbed := lookupEmbeddingOrNull(token.Form, ivr.wordEmbeds)
		tagEmbed := lookupEmbeddingOrNull(token.PosTag, ivr.tagEmbeds)

		copy(data[idx*embeddingsSize:], wordEmbed)
		copy(data[(idx*embeddingsSize)+len(wordEmbed):], tagEmbed)
	}

	return Input{
		data:      data,
		timesteps: len(sentence),
	}
}

func (ivr *Realizer) InputLen() int {
	return ivr.wordEmbeds.EmbeddingSize() + ivr.tagEmbeds.EmbeddingSize()
}

func lookupEmbeddingOrNull(lookup func() (string, bool), embeddings *go2vec.Embeddings) []float32 {
	val := "_"
	if realVal, ok := lookup(); ok {
		val = realVal
	}

	if embedding, ok := embeddings.Embedding(val); ok {
		return embedding
	}

	return make([]float32, embeddings.EmbeddingSize())
}
