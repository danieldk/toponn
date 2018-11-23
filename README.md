# toponn

**toponn** is a topological field tagger using recurrent neural networks.
The original Go version can be found in the
[oldgo](https://github.com/danieldk/toponn/tree/oldgo) branch.

## Building

Building toponn has the following requirements:

* A reasonably [modern Rust compiler](https://rustup.rs).
* Tensorflow built as a dynamic library (the Python module is **only** to construct/write the graph).

## macOS

Install the dependencies using Homebrew:

~~~bash
$ brew install rustup-init libtensorflow
# Install/configure the Rust toolchain.
$ rustup-init
~~~

Then compile and install toponn:

~~~bash
$ cd toponn
$ cargo install --path toponn-utils
~~~

toponn should then be installed in ~/.cargo/bin/toponn-{tag,train,server}

## References

If you use toponn in a scientific publication, please cite:

* Daniël de Kok and Erhard Hinrichs, *Transition-based dependency parsing with topological fields*, Proceedings of ACL 2016, Berlin, Germany
