// Copyright 2015 The dparnn Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package common

import (
	"io"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type TopoNNConfig struct {
	TensorFlow TensorFlow
	Data       Data
	Embeddings Embeddings
	Fields     string
}

type Data struct {
	Timesteps int
	BatchSize int `toml:"batch_size"`
}

type TensorFlow struct {
	GPUMemoryFraction float64 `toml:"gpu_mem_frac"`
	Graph             string
}

type Embeddings struct {
	Word Embedding
	Tag  Embedding
}

type Embedding struct {
	Filename       string
	Normalize      bool
	NormalizeInput bool
}

func defaultConfiguration() *TopoNNConfig {
	return &TopoNNConfig{
		Data: Data{
			Timesteps: 150,
			BatchSize: 128,
		},
		TensorFlow: TensorFlow{
			GPUMemoryFraction: 0.3,
			Graph:             "graph.binaryproto",
		},
		Embeddings: Embeddings{
			Word: Embedding{
				Filename:       "word-vectors.bin",
				Normalize:      true,
				NormalizeInput: true,
			},
			Tag: Embedding{
				Filename:       "tag-vectors.bin",
				Normalize:      true,
				NormalizeInput: true,
			},
		},
		Fields: "fields.labels",
	}
}

func ParseConfig(reader io.Reader) (*TopoNNConfig, error) {
	config := defaultConfiguration()
	_, err := toml.DecodeReader(reader, config)
	return config, err
}

func MustReadConfig(filename string) *TopoNNConfig {
	f, err := os.Open(filename)
	ExitIfError("Error opening configuration file: ", err)
	defer f.Close()
	config, err := ParseConfig(f)
	ExitIfError("Error parsing configuration file: ", err)

	config.Embeddings.Word.Filename = relToConfig(filename, config.Embeddings.Word.Filename)
	config.Embeddings.Tag.Filename = relToConfig(filename, config.Embeddings.Tag.Filename)
	config.Fields = relToConfig(filename, config.Fields)
	config.TensorFlow.Graph = relToConfig(filename, config.TensorFlow.Graph)

	return config
}

// Return the path of a file, relative to the directory of
// the configuration file, unless the path is absolute.
func relToConfig(configPath, filePath string) string {
	if len(filePath) == 0 {
		return filePath
	}

	if filepath.IsAbs(filePath) {
		return filePath
	}

	return filepath.Join(filepath.Dir(configPath), filePath)
}
