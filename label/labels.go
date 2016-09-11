// Copyright 2015 The dpar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package label

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// A label numberer creates a bijection between (string-based)
// features and numbers.
type LabelNumberer struct {
	labelNumbers map[string]int
	labels       []string
}

func NewLabelNumberer() LabelNumberer {
	return LabelNumberer{make(map[string]int), make([]string, 0)}
}

func (l *LabelNumberer) Number(label string) int {
	idx, ok := l.labelNumbers[label]

	if !ok {
		idx = len(l.labelNumbers) + 1
		l.labelNumbers[label] = idx
		l.labels = append(l.labels, label)
	}

	return idx
}

func (l LabelNumberer) Label(number int) (string, bool) {
	if number < 1 || number > len(l.labels) {
		return "", false
	}

	return l.labels[number-1], true
}

func (l LabelNumberer) Size() int {
	return len(l.labels)
}

func (l *LabelNumberer) Read(reader io.Reader) error {
	var labels []string
	bufReader := bufio.NewReader(reader)

	eof := false
	for !eof {
		line, err := bufReader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			}

			eof = true
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		labels = append(labels, strings.TrimSpace(line))
	}

	numbers := make(map[string]int)
	for idx, label := range labels {
		numbers[label] = idx + 1
	}

	l.labels = labels
	l.labelNumbers = numbers

	return nil
}

func (l *LabelNumberer) WriteLabelNumberer(writer io.Writer) error {
	for _, label := range l.labels {
		fmt.Fprintln(writer, label)
	}

	return nil
}
